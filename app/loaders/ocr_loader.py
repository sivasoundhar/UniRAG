"""
Load an image file into a LangChain `Document` via OCR.

Returns the exact same shape as app/loaders/file_loader.py (a list of
Documents with a `source` metadata key) so that downstream chunking code
has one contract to satisfy, regardless of whether the text originally
came from a PDF or a scanned photo of a whiteboard.
"""

from pathlib import Path

import pytesseract
from langchain_core.documents import Document
from PIL import Image

from app.config.settings import settings

# Tesseract's binary location isn't on PATH by default on every OS/install,
# so we point pytesseract at it explicitly via settings rather than relying
# on the environment being set up correctly.
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def load_image(file_path: str) -> list[Document]:
    """
    Takes: path to an image file (.png/.jpg/.jpeg/.tiff/.bmp).
    Returns: a single-element list containing one Document whose
    page_content is the OCR'd text and whose metadata["source"] is the
    filename.
    Use this: as the image counterpart to load_file() in file_loader.py,
    anywhere the ingest pipeline needs to accept a scanned document.

    Fails loudly if the file is missing, has an unsupported extension, or
    OCR extracts nothing — an empty string silently becoming an empty
    chunk downstream is a much harder bug to trace than an error raised
    right here at the source of the text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    extension = path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image type '{extension}' for {path.name}. "
            f"Supported types: {supported}"
        )

    text = pytesseract.image_to_string(Image.open(path)).strip()
    if not text:
        raise ValueError(
            f"OCR extracted no text from {path.name} — check image quality "
            "or that Tesseract is installed and TESSERACT_CMD is correct."
        )

    return [Document(page_content=text, metadata={"source": path.name})]


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
        print(f"  source={doc.metadata['source']!r} text={doc.page_content!r}")

    Path(tmp_path).unlink()
