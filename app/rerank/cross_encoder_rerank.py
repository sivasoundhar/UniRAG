"""
Rerank a small candidate set by scoring (query, passage) pairs together.

Why this works: dense retrieval (BGEEmbedder) is a bi-encoder — it embeds
the query and each passage *separately*, so similarity is a dot product
between two vectors that were never compared to each other during
encoding. That's what makes it fast enough to search a whole corpus (every
passage is embedded once, up front, and reused for every query). A
cross-encoder instead feeds the query and passage into the model *together*
as one input, so attention can run between every query token and every
passage token — it can catch "these share vocabulary but answer different
questions" in a way two independently-computed vectors can't. The tradeoff
is exactly its strength: one forward pass per (query, passage) pair means
it cannot run over an entire corpus, only over the small top-k that
retrieval + RRF has already narrowed things down to.

Two natural follow-up questions:
1. "Why not just use the cross-encoder for retrieval directly, and skip
   the embedding step?" — because it doesn't scale: reranking 5 or 10
   candidates costs 5-10 forward passes, but scoring a whole corpus this
   way would cost one forward pass per document per query, which is
   untenable past a few thousand documents. Bi-encoder retrieval exists
   specifically to produce a short list a cross-encoder can afford to
   re-score.
2. "The cross-encoder disagrees with RRF's ranking — which do you trust?"
   — the cross-encoder, for final ordering. RRF's job was recall (don't
   lose the right document among the candidates); the cross-encoder's job
   is precision on the exact query at hand, using a model shape suited to
   fine-grained relevance judgments rather than coarse ranking of many
   candidates from two very different scoring methods.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config.settings import settings

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load the cross-encoder once per process, not once per call."""
    global _model
    if _model is None:
        _model = CrossEncoder(settings.cross_encoder_model)
    return _model


def rerank(
    query: str,
    documents: list[Document],
    top_n: int = settings.rerank_top_n,
) -> list[tuple[Document, float]]:
    """
    Takes: the user query and a small candidate list of Documents (e.g.
    RRF-fused retrieval output — expected to already be narrowed to
    roughly retrieval_top_k candidates, not the whole corpus).
    Returns: the top_n (Document, score) pairs, ordered by cross-encoder
    relevance score descending. Score is an unbounded real number (a raw
    logit, not a probability) — only its ordering matters here, it isn't
    meant to be compared across different queries.
    Use this: as the final step before handing context to the LLM, after
    retrieval and fusion have already produced a short candidate list.
    """
    if not documents:
        return []

    model = _get_model()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_n]]


if __name__ == "__main__":
    # Tiny self-test: a deliberately-suboptimal input order (the truly
    # relevant document placed last) should come out on top after rerank.
    query = "How does reciprocal rank fusion combine BM25 and dense search?"
    candidates = [
        Document(page_content="Tesseract is an open-source OCR engine for reading text out of images.", metadata={"source": "ocr.txt"}),
        Document(page_content="ChromaDB persists embeddings to disk so the index survives a restart.", metadata={"source": "chroma.txt"}),
        Document(page_content="RRF fuses BM25 and dense retrieval by summing reciprocal ranks, not raw scores.", metadata={"source": "rrf.txt"}),
    ]

    print("Input order:")
    for doc in candidates:
        print(f"  source={doc.metadata['source']!r}")

    results = rerank(query, candidates, top_n=3)
    print("\nReranked order:")
    for doc, score in results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r}")

    assert results[0][0].metadata["source"] == "rrf.txt", "the relevant doc should rank first after rerank"
    print("\nOK: relevant document moved to rank 1 after reranking.")
