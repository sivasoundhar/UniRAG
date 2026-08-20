"""
Load a document from disk into LangChain `Document` objects.

This is the single entry point every downstream module (chunkers, the
upload endpoint) uses to turn a file on disk into text — they never touch
a loader class directly. That indirection is the whole point: adding a new
file type later means adding one branch here, not touching chunking or the
API layer.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

# Extension -> loader class. A dict instead of if/elif so adding a type is
# a one-line change and the supported-type list is visible at a glance.
_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_file(file_path: str) -> list[Document]:
    """
    Takes: path to a .pdf, .txt, or .docx file on disk.
    Returns: a list of LangChain Documents (one per page for PDFs, one for
    the whole file otherwise), each with `metadata["source"]` set to the
    filename so a chunk can always be traced back to where it came from.
    Use this: as the first step of the ingest pipeline, before chunking.

    Fails loudly on an unsupported extension or a missing file — silently
    returning an empty list here would make an empty index look identical
    to "nothing to index," which is a much harder bug to find later.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    extension = path.suffix.lower()
    loader_cls = _LOADERS.get(extension)
    if loader_cls is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported file type '{extension}' for {path.name}. "
            f"Supported types: {supported}"
        )

    documents = loader_cls(str(path)).load()

    # PyPDFLoader/TextLoader already set metadata["source"] to the full
    # path passed in; normalise it to just the filename so metadata stays
    # stable regardless of where the upload was temporarily stored on disk.
    for doc in documents:
        doc.metadata["source"] = path.name

    return documents


if __name__ == "__main__":
    # Tiny self-test: create a throwaway .txt file and load it back.
    # PDF/image loading gets exercised in the pytest suite instead, once
    # sample files exist under data/uploads/.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("UniRAG is a reusable AI knowledge core.\n")
        f.write("It fuses BM25 and dense retrieval with RRF.\n")
        tmp_path = f.name

    docs = load_file(tmp_path)
    print(f"Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"  source={doc.metadata['source']!r} text={doc.page_content!r}")

    Path(tmp_path).unlink()
