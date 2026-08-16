"""
Load an image file into a LangChain `Document` via OCR, falling back to a
vision model's description when OCR finds no text at all.

Returns the exact same shape as app/loaders/file_loader.py (a list of
Documents with a `source` metadata key) so that downstream chunking code
has one contract to satisfy, regardless of whether the text originally
came from a PDF, a scanned photo of a whiteboard, or a photo/scan with no
text in it at all (an X-ray, a diagram, a CAE render).

Two genuinely different failure modes used to collapse into the same
"OCR found nothing" error before the vision fallback existed: a *bad scan*
of real text (fixable by rescanning) and an image that was *never text in
the first place* (not fixable by rescanning — there's nothing to OCR,
ever). The vision fallback only helps with the second case, and can't tell
the two apart on its own — see load_image's docstring for how it's told
apart in practice (it isn't; vision is just always tried second).
"""

from pathlib import Path

import pytesseract
from langchain_core.documents import Document
from PIL import Image

from app.config.settings import settings
from app.llm.llm_gateway import get_vision_completion

# Tesseract's binary location isn't on PATH by default on every OS/install,
# so we point pytesseract at it explicitly via settings rather than relying
# on the environment being set up correctly.
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

_VISION_PROMPT = (
    "Describe this image in detail for a search index — what it shows, any "
    "objects, structures, labels, numbers, or notable features. If it's a "
    "medical, engineering, or technical image, describe what a domain "
    "expert would notice. Be specific and factual; do not speculate beyond "
    "what's visible. If a person appears in the image, describe their "
    "visible appearance and context (clothing, setting, pose, expression) "
    "— do NOT assert who they are by name unless a name is clearly and "
    "legibly written/captioned within the image itself. You cannot "
    "reliably recognize faces, and confidently stating a wrong name is "
    "worse than describing them without one — a wrong name gets indexed "
    "as if it were a verified fact, and answers questions about the wrong "
    "person entirely."
)


def load_image(file_path: str, original_filename: str | None = None) -> list[Document]:
    """
    Takes: path to an image file (.png/.jpg/.jpeg/.tiff/.bmp) on disk, and
    optionally the name it should be known by if that differs from the
    file on disk (e.g. app/main.py's upload route saves to a UUID-prefixed
    temp path to avoid collisions, but wants the original filename used
    everywhere it matters). Defaults to the on-disk filename when omitted.
    Returns: a single-element list containing one Document whose
    page_content is either the OCR'd text (if any was found) or a vision
    model's description of the image (if OCR found none), and whose
    metadata["source"] is the filename. metadata["extraction_method"] is
    "ocr" or "vision" so it's checkable afterward which path actually ran,
    not just assumed.
    Use this: as the image counterpart to load_file() in file_loader.py,
    anywhere the ingest pipeline needs to accept a scanned document *or* a
    photo/scan with no text in it (an X-ray, a diagram, a photo of
    equipment) — OCR alone can only ever handle the former.

    Fails loudly if the file is missing, has an unsupported extension, or
    both OCR and the vision fallback come up empty — an empty string
    silently becoming an empty chunk downstream is a much harder bug to
    trace than an error raised right here at the source of the text.

    original_filename existing as a real parameter (not something the
    caller patches into metadata afterward) is not cosmetic: this file's
    vision path writes the filename directly into page_content (see
    below), and that has to be the meaningful name at the point it's
    written, not fixed up later — metadata can be overwritten after the
    fact, but text already baked into a stored chunk can't be. A caller
    that upload-saved to a UUID-prefixed temp path and only patched
    metadata["source"] afterward would otherwise permanently index the
    UUID as the "Filename:" line instead of the real name — this bit a
    real upload during testing (source name in metadata was correct, but
    the actual indexed text still said "Filename: 8fda1f12-...jpg").
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    display_name = original_filename or path.name

    extension = path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image type '{extension}' for {path.name}. "
            f"Supported types: {supported}"
        )

    text = pytesseract.image_to_string(Image.open(path)).strip()
    if text:
        return [Document(page_content=text, metadata={"source": display_name, "extraction_method": "ocr"})]

    # OCR found nothing — could be a bad scan of real text, but far more
    # often (in practice, X-rays being the concrete case that motivated
    # this) it's an image that was never text to begin with. Try describing
    # it instead of failing immediately.
    try:
        vision_result = get_vision_completion(str(path), _VISION_PROMPT)
    except Exception as e:
        raise ValueError(
            f"OCR extracted no text from {display_name}, and the vision fallback "
            f"also failed ({e}). Check image quality, that Tesseract is "
            "installed and TESSERACT_CMD is correct, and that GROQ_API_KEY "
            "is set for the vision fallback."
        ) from e

    if not vision_result.text:
        raise ValueError(
            f"OCR extracted no text from {display_name}, and the vision model "
            "returned an empty description — check image quality."
        )

    # The vision prompt deliberately won't assert a specific person's
    # identity it can't actually verify from pixels (see _VISION_PROMPT) —
    # necessary so it stops confidently guessing WRONG names, but the
    # consequence is a genuinely correct name never appears in the
    # description either, so a later question using that name ("explain
    # about saipallavi") can't match anything in the text. The uploader's
    # own filename is a much more reliable signal than a pixel-guess (most
    # people name their files accurately) and previously never made it
    # into the indexed content at all — only metadata["source"], which
    # generate_node's context never includes. Prepending it here means
    # retrieval and the final answer can both use it, while the actual
    # visual description underneath stays honest about what it can and
    # can't confirm from the image itself.
    page_content = f"Filename: {display_name}\n\n{vision_result.text}"

    return [
        Document(
            page_content=page_content,
            metadata={"source": display_name, "extraction_method": "vision", "vision_model": vision_result.model},
        )
    ]


if __name__ == "__main__":
    # Tiny self-test: render a line of text into a fresh PNG with Pillow
    # (no external sample image needed) and OCR it back.
    import tempfile

    from PIL import ImageDraw

    img = Image.new("RGB", (400, 60), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "UniRAG OCR self-test", fill="black")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        tmp_path = f.name

    docs = load_image(tmp_path)
    print(f"Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"  source={doc.metadata['source']!r} extraction_method={doc.metadata['extraction_method']!r} text={doc.page_content!r}")
    assert docs[0].metadata["extraction_method"] == "ocr"
    print("OK: an image with real text uses the OCR path.")

    Path(tmp_path).unlink()

    # Now the vision fallback: a genuinely blank image (nothing for OCR to
    # find, same as the real X-ray case that motivated this), with the
    # vision call itself mocked — deterministic and doesn't depend on
    # Groq being reachable (it hasn't been, reliably, all session).
    from unittest.mock import patch

    from app.llm.llm_gateway import Completion

    blank_img = Image.new("RGB", (100, 100), color="white")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        blank_img.save(f.name)
        blank_path = f.name

    print("\nVerifying vision fallback (blank image, OCR finds nothing)...")
    with patch(f"{__name__}.get_vision_completion") as mock_vision:
        mock_vision.return_value = Completion(
            text="A blank white image with no visible content.",
            model="qwen/qwen3.6-27b",
            provider="groq",
        )
        docs = load_image(blank_path)

    print(f"Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"  source={doc.metadata['source']!r} extraction_method={doc.metadata['extraction_method']!r} vision_model={doc.metadata.get('vision_model')!r} text={doc.page_content!r}")
    assert docs[0].metadata["extraction_method"] == "vision"
    assert docs[0].metadata["vision_model"] == "qwen/qwen3.6-27b"
    assert mock_vision.called
    # Filename gets prepended to page_content (not just left in metadata) so
    # retrieval/generation can actually use it — see load_image's comment
    # on why: the vision description alone deliberately won't assert an
    # identity it can't verify, so the filename is often the only reliable
    # name-shaped signal a later question could match against.
    assert docs[0].page_content.startswith(f"Filename: {Path(blank_path).name}\n\n")
    print("OK: an image with no text (OCR finds nothing) falls back to the vision model, filename included in content.")

    Path(blank_path).unlink()
