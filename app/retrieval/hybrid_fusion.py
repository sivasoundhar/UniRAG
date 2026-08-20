"""
Reciprocal Rank Fusion (RRF): merge dense and BM25 rankings into one list.

Why rank, not score: dense similarity and BM25 scores live on unrelated
scales (cosine similarity in [-1, 1] vs. an unbounded BM25 sum) and have
different distributions even for "obviously relevant" results. Normalising
either one onto a shared scale is guesswork — there's no principled way to
say a BM25 score of 8.2 "equals" a cosine score of 0.61. RRF sidesteps the
whole problem by throwing scores away and using only each document's
*position* in each ranked list, which is directly comparable across
retrievers by construction.

The formula, for a document d found by a set of rankers R:

    RRF(d) = sum over r in R of  1 / (k + rank_r(d))

- rank_r(d) is d's 1-indexed position in ranker r's list (1 = top result).
- k is a smoothing constant (settings.rrf_k, 60 here — the value used in
  the original Cormack et al. 2009 paper). Without k, rank 1 vs rank 2
  would differ by a factor of 2 (1/1 vs 1/2); with k=60, they differ by
  about 1.6% (1/61 vs 1/62) — the top of any single list stops dominating
  the fused result, which matters because BM25 and dense retrieval are
  about equally likely to be "the one that's right" for a given query.
- A document missing from a ranker's list contributes 0 for that ranker —
  it isn't penalised beyond simply not getting that ranker's vote.

A note on the self-test below: it does *not* show dense retrieval "failing"
on a keyword query. BGE-small turns out to be quite good at matching literal
IDs/codes on its own (subword tokenization gives it real signal there), so on
a small clean corpus both retrievers usually agree on the #1 result — that's
a genuinely useful sanity check in its own right (two independent methods
confirming the same answer). Where fusion visibly earns its keep here is the
*runner-up* ordering: dense and BM25 rank the next few candidates
differently, based on different evidence, and RRF produces a third ordering
that reflects both rather than picking one retriever to trust blindly.

Two natural follow-up questions:
1. "Why not just average the normalised scores instead?" — because
   normalising (e.g. min-max per query) makes the fused score sensitive to
   the score *distribution* of that one query's result set, not just
   relevance; a query where BM25 scores are all clustered together would
   get artificially inflated weight relative to one where they're spread
   out. Rank position doesn't have this problem — it's already query-
   relative by definition.
2. "What happens if one retriever returns garbage for a query?" — RRF is
   robust to this by design: a bad ranker's rank-1 result only contributes
   1/(k+1) to that one document, the same weight a good ranker's rank-1
   result gets. It can't be down-weighted automatically, but it also can't
   dominate just by being confidently wrong (a raw-score approach could,
   if that ranker's score scale runs higher).
"""

from langchain_core.documents import Document

from app.config.settings import settings


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = settings.rrf_k,
) -> list[tuple[Document, float]]:
    """
    Takes: one or more ranked (Document, score) lists (e.g. dense results,
    BM25 results) — the input score in each pair is ignored, only the
    position in its list matters. Documents are identified by
    metadata["chunk_id"], so the same chunk found by multiple rankers
    accumulates a contribution from each.
    Returns: a single list of (Document, rrf_score) pairs, sorted by
    rrf_score descending — chunks found (and ranked highly) by more than
    one retriever rise to the top.
    Use this: as the merge step between dense and BM25 retrieval, before
    reranking.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            chunk_id = doc.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            documents.setdefault(chunk_id, doc)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(documents[chunk_id], score) for chunk_id, score in fused]


if __name__ == "__main__":
    # Self-test demo: one document contains the literal ID from the
    # query (the true answer); four more are near-verbatim echoes of the
    # query itself with the ID removed — maximum semantic similarity, zero
    # keyword overlap. Both retrievers correctly agree on the #1 result
    # (a real confidence signal), then diverge on how to rank the four
    # near-duplicate runner-ups — that divergence, and RRF's reconciliation
    # of it, is the actual mechanism being demonstrated.
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.vector_store import VectorStore

    store = VectorStore()
    store.add_documents([
        Document(page_content="Ticket ref Zx9Qv2Lm7Rb was closed by support yesterday.", metadata={"source": "target.txt"}),
        Document(page_content="Was this support ticket closed by our team?", metadata={"source": "echo1.txt"}),
        Document(page_content="Has this particular ticket been closed yet?", metadata={"source": "echo2.txt"}),
        Document(page_content="Is the ticket closed now?", metadata={"source": "echo3.txt"}),
        Document(page_content="Was the support ticket already closed?", metadata={"source": "echo4.txt"}),
    ])

    bm25 = BM25Retriever()
    bm25.build(store.get_all_documents())

    query = "Was ticket Zx9Qv2Lm7Rb closed?"
    dense_results = store.similarity_search(query, k=5)
    bm25_results = bm25.search(query, k=5)
    fused_results = reciprocal_rank_fusion([dense_results, bm25_results])

    print(f"Query: {query!r}\n")
    print("Dense-only ranking:")
    for doc, score in dense_results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r}")
    print("\nBM25-only ranking:")
    for doc, score in bm25_results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r}")
    print("\nRRF-fused ranking:")
    for doc, score in fused_results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r}")

    assert dense_results[0][0].metadata["source"] == "target.txt"
    assert bm25_results[0][0].metadata["source"] == "target.txt"
    assert fused_results[0][0].metadata["source"] == "target.txt"
    print("\nOK: dense, BM25, and RRF all agree on the correct top result;")
    print("note the runner-up order differs across all three above.")
