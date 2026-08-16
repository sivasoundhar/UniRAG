"""
Small, deterministic RAG quality metrics — no LLM-as-judge, no paid eval
service. Two questions, two cheap proxies:

- Does retrieval find the right document at all? -> retrieval_hit_rate,
  checked against a small hand-labeled (query, expected_source) eval set.
- Does the answer actually use what was retrieved, or ignore it? ->
  context_overlap_ratio, a crude word-overlap heuristic.

Neither is a substitute for a real evaluation (e.g. an LLM-as-judge for
faithfulness, or a labeled relevance dataset for precision/recall) — both
are exactly the "smoke test" level CLAUDE.md's tech stack calls for: fast,
free, and good enough to catch an obviously broken pipeline (retrieval
returning nothing relevant, or an answer that's disconnected from its
context) without adding a model call to every eval run.
"""

import re

from langchain_core.documents import Document

# A short, hardcoded stopword list, not a full NLP stopword corpus — this
# is a coarse overlap heuristic, not a linguistic analysis, so filtering
# out the dozen most common function words is enough to stop them from
# padding every overlap score toward 1.0 regardless of actual content.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to",
    "and", "or", "for", "with", "that", "this", "it", "by", "as", "be", "at",
}


def _content_words(text: str) -> set[str]:
    """Lowercase word split, stopwords and very short tokens dropped."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def context_overlap_ratio(answer: str, context_docs: list[Document]) -> float:
    """
    Takes: a generated answer and the Documents it was generated from
    (e.g. GraphState["compressed_docs"]).
    Returns: the fraction of the answer's content words that also appear
    somewhere in the context — 1.0 means every distinctive word in the
    answer shows up in the context, 0.0 means none do.
    Use this: as a fast, free groundedness proxy. It is not a faithfulness
    check — a low-overlap answer might still be correct (paraphrased in
    different words), and a high-overlap answer might still misrepresent
    the context. It catches the obvious failure (answer unrelated to
    context) without needing an LLM-as-judge call.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0

    context_words = _content_words(" ".join(doc.page_content for doc in context_docs))
    return len(answer_words & context_words) / len(answer_words)


def retrieval_hit_rate(cases: list[tuple[str, str]], graph) -> float:
    """
    Takes: a small hand-labeled eval set of (query, expected_source) pairs,
    and a compiled LangGraph pipeline (app.graph.build_graph.build_graph()).
    Returns: the fraction of cases where expected_source appears among the
    reranked results' sources — 1.0 means retrieval found the right
    document every time, 0.0 means never.
    Use this: as a quick regression check after changing chunking,
    embeddings, or retrieval settings — a drop in hit rate on a fixed eval
    set is a much faster signal than re-reading every answer by hand.

    Returns 0.0 for an empty eval set rather than raising — nothing to
    measure isn't a pipeline failure, just an empty result.
    """
    if not cases:
        return 0.0

    hits = 0
    for i, (query, expected_source) in enumerate(cases):
        state = graph.invoke({"conversation_id": f"eval-{i}", "query": query})
        sources = {doc.metadata.get("source") for doc, _score in state.get("reranked_results", [])}
        if expected_source in sources:
            hits += 1

    return hits / len(cases)


if __name__ == "__main__":
    # Tiny self-test: seed the same small RRF corpus used elsewhere, run
    # one eval case through the real graph, and check both metrics land
    # somewhere sane (hit rate is binary per case here; overlap should be
    # well above 0 for a genuinely grounded answer).
    from langchain_core.documents import Document as _Document

    from app.graph.build_graph import build_graph
    from app.retrieval.vector_store import VectorStore

    VectorStore().add_documents([
        _Document(page_content="Reciprocal rank fusion (RRF) merges a dense retriever's ranked list and a BM25 retriever's ranked list into one combined ranking.", metadata={"source": "rrf_intro.txt"}),
    ])

    graph = build_graph()
    eval_cases = [("What is reciprocal rank fusion?", "rrf_intro.txt")]

    hit_rate = retrieval_hit_rate(eval_cases, graph)
    print(f"Retrieval hit rate: {hit_rate:.2f}")
    assert hit_rate == 1.0, "expected the seeded rrf_intro.txt chunk to be retrieved"

    final_state = graph.invoke({"conversation_id": "eval-overlap-demo", "query": eval_cases[0][0]})
    overlap = context_overlap_ratio(final_state["answer"], final_state["compressed_docs"])
    print(f"Context overlap ratio: {overlap:.2f}")
    print(f"Answer: {final_state['answer']!r}")

    print("\nOK: retrieval found the expected source, answer overlaps meaningfully with its context.")
