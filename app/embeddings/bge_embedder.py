"""
Wrap BAAI/bge-small-en-v1.5 for indexing and querying Chroma.

This is a separate call site from app/chunking/semantic_chunker.py, which
loads its own SentenceTransformer to find sentence-boundary breakpoints at
ingest time. That module runs once per document, on raw sentences, before a
chunk exists. This one runs on finished chunks (for indexing) and on user
queries (for retrieval) — different lifecycle, same underlying model, so the
duplication between the two modules is intentional rather than an oversight.

The one thing that matters here and doesn't matter there: BGE models are
trained asymmetrically. The query side needs an instruction prefix telling
the model "this text is a search query"; the passage side gets no prefix at
all. Mixing this up (or dropping it) still runs without error — it just
quietly hurts recall, which is the kind of bug that's easy to ship unnoticed.
"""

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the embedding model once per process, not once per call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


class BGEEmbedder:
    """
    Takes: nothing at construction (model is loaded lazily on first use).
    Returns: an object exposing embed_documents / embed_query.
    Use this: anywhere text needs to become a vector for Chroma — indexing
    in app.retrieval.vector_store, and query embedding at search time.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Takes: a list of chunk texts to index.
        Returns: one normalized embedding vector per text, no instruction
        prefix — BGE passages are embedded plain.
        Use this: when writing chunks into the vector store.
        """
        model = _get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Takes: a single user query string.
        Returns: a normalized embedding vector, with BGE's query instruction
        prefix applied so the query lands in the same trained distribution
        as the passages it's being compared against.
        Use this: at search time, never at indexing time.
        """
        model = _get_model()
        vector = model.encode(settings.bge_query_instruction + text, normalize_embeddings=True)
        return vector.tolist()


if __name__ == "__main__":
    # Tiny self-test: a query and its true match should be far more similar
    # than the query and an unrelated sentence — proves both the model and
    # the query-instruction prefix are wired correctly.
    import numpy as np

    embedder = BGEEmbedder()
    query = "How does reciprocal rank fusion combine BM25 and dense search?"
    related = "RRF fuses BM25 and dense retrieval by summing reciprocal ranks, not raw scores."
    unrelated = "Tesseract is an open-source OCR engine for reading text out of images."

    query_vec = np.array(embedder.embed_query(query))
    doc_vecs = np.array(embedder.embed_documents([related, unrelated]))

    similarities = doc_vecs @ query_vec  # vectors are normalized, so dot product == cosine similarity
    print(f"cosine(query, related)   = {similarities[0]:.4f}")
    print(f"cosine(query, unrelated) = {similarities[1]:.4f}")
    assert similarities[0] > similarities[1], "related sentence should score higher than unrelated"
    print("OK: related sentence scored higher than unrelated.")
