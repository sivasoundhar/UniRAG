"""
Split Documents on meaning instead of size: walk sentence by sentence and
cut wherever two consecutive sentences are less alike than most
sentence-pairs in the document.

Where app/chunking/recursive_chunker.py cuts at a fixed character count
regardless of what's being said, this catches topic boundaries that fall
mid-paragraph — the tradeoff is one embedding call per sentence, so it's
slower and only worth it for documents where topics shift abruptly (e.g.
a spec that jumps from "auth" to "billing" inside one paragraph).

Note on the embedding call below: this module loads its own
SentenceTransformer instance rather than importing app/embeddings/bge_embedder.py.
That module wraps the same model for indexing
into Chroma. Splitting sentences to find breakpoints and embedding chunks
for retrieval are different call sites with different lifecycles (this one
runs once at ingest time on raw sentences, that one runs on finished
chunks), so the duplication is intentional for now rather than an oversight.
If it becomes annoying to keep two embedding call sites in sync, the fix is
to have this module take an embedder as a parameter instead of owning one.
"""

import re

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.config.settings import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the embedding model once per process, not once per call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Takes: an (n_sentences, dim) array of embeddings.
    Returns: an (n_sentences - 1,) array where entry i is 1 - cosine
    similarity between sentence i and sentence i+1 — small when adjacent
    sentences are topically similar, large when the topic jumps.
    """
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1 - similarities


def chunk_semantic(documents: list[Document]) -> list[Document]:
    """
    Takes: a list of Documents (e.g. from app.loaders.file_loader.load_file).
    Returns: a list of Documents grouped by topic — each chunk is a run of
    consecutive sentences with no large meaning-jump between them.
    Metadata (including "source") is copied onto every chunk.
    Use this: as the alternative chunker for documents with abrupt topic
    shifts; app.chunking.recursive_chunker.chunk_recursive is the default.

    A document with fewer than 2 sentences has no boundary to find, so it
    is returned as a single chunk rather than raising — this is normal
    input (e.g. a short OCR'd label), not an error condition.
    """
    model = _get_model()
    chunks: list[Document] = []

    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if len(sentences) < 2:
            chunks.append(doc)
            continue

        embeddings = model.encode(sentences)
        distances = _cosine_distances(embeddings)

        # A breakpoint is a distance further out than most others in this
        # document — using a percentile (not a fixed threshold) means the
        # cut is relative to how varied this specific document is, so a
        # uniformly technical doc and a uniformly casual one both get
        # sensible splits without retuning a magic number per document.
        threshold = np.percentile(distances, settings.semantic_breakpoint_percentile)
        breakpoints = {i + 1 for i, d in enumerate(distances) if d > threshold}

        start = 0
        for i in range(1, len(sentences) + 1):
            if i in breakpoints or i == len(sentences):
                text = " ".join(sentences[start:i])
                chunks.append(Document(page_content=text, metadata=dict(doc.metadata)))
                start = i

    return chunks


if __name__ == "__main__":
    # Tiny self-test: two distinct topics back to back should produce >= 2
    # chunks, with the split falling between the topics, not inside one.
    sample_text = (
        "UniRAG fuses BM25 and dense retrieval using RRF. "
        "RRF ranks results by reciprocal rank, not raw score. "
        "This avoids having to normalise BM25 and cosine scores. "
        "Tesseract is an open-source OCR engine maintained by Google. "
        "It reads text out of scanned images and photos. "
        "OCR quality depends heavily on image resolution and contrast."
    )
    sample_doc = Document(page_content=sample_text, metadata={"source": "demo.txt"})

    chunks = chunk_semantic([sample_doc])
    print(f"Produced {len(chunks)} chunk(s) from 1 document:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] {chunk.page_content!r}")
