"""
Split Documents into fixed-size, overlapping chunks along natural text
boundaries (paragraph -> sentence -> word -> char, in that order).

This is the default, cheap chunking strategy: fast, deterministic, no model
calls. It's a reasonable baseline for most text. app/chunking/semantic_chunker.py
is the more expensive alternative that splits on meaning instead of size —
use that when a document has abrupt topic shifts that fixed-size windows
would cut through.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings


def chunk_recursive(documents: list[Document]) -> list[Document]:
    """
    Takes: a list of Documents (e.g. from app.loaders.file_loader.load_file).
    Returns: a list of smaller Documents, each <= settings.chunk_size chars,
    with settings.chunk_overlap chars shared between consecutive chunks.
    Metadata (including "source") is copied onto every chunk so a chunk
    can always be traced back to its parent document.
    Use this: as the default chunker for the ingest pipeline.
    """
    # The separator list is tried in order: split on paragraphs first, and
    # only fall back to sentences/words/chars if a paragraph is still
    # bigger than chunk_size. This is why "recursive" — it recurses down
    # the list until pieces are small enough.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Tiny self-test: a paragraph long enough (> settings.chunk_size chars)
    # to force at least one split.
    sample_text = (
        "UniRAG is a reusable AI knowledge core built as the retrieval "
        "backbone for downstream copilots. It combines dense and "
        "sparse retrieval fused with RRF, then reranks with a cross-encoder. "
        "Query rewriting and expansion improve recall before retrieval even "
        "runs. Guardrails and evaluation scaffolding round out the pipeline. "
        "Each module is meant to be readable top to bottom in one sitting, "
        "with heavy inline comments explaining why a default was chosen, "
        "not what the code obviously already says. Chunk size, overlap, "
        "and every other tunable live in one settings file, so a question "
        "about any magic number has a one-line answer."
    )
    sample_doc = Document(page_content=sample_text, metadata={"source": "demo.txt"})

    chunks = chunk_recursive([sample_doc])
    print(f"Produced {len(chunks)} chunk(s) from 1 document:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] ({len(chunk.page_content)} chars) {chunk.page_content!r}")
