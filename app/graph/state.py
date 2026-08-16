"""
Shared state that flows through the LangGraph pipeline (see build_graph.py).

Kept as a flat TypedDict with default (replace) merge semantics, not
Annotated reducers — this graph is a single linear path (see the ASCII
diagram in build_graph.py's docstring) where no two nodes ever write the
same key at the same time, so LangGraph's default "last write wins" merge
is exactly correct and a custom reducer would be solving a problem this
graph doesn't have.
"""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Takes: nothing directly — this is a schema, not a constructor. A caller
    starts a graph run by providing conversation_id and query; every other
    key is filled in by a node as the state passes through it.
    Returns: n/a.
    Use this: as the state_schema passed to langgraph.graph.StateGraph.

    total=False because only conversation_id/query are known up front —
    every other field is genuinely optional until its node has run, and
    marking them required would just force every node to fill in
    placeholder values for keys it hasn't computed yet.
    """

    conversation_id: str
    query: str
    rewritten_query: str
    # Alternate phrasings of rewritten_query (app/query/expansion.py), used
    # alongside it so retrieval isn't limited to one exact wording — e.g.
    # "LLM" vs "language model" vs "chatbot". Can be empty: expand_node
    # falls back to [] rather than failing the turn if the LLM call errors,
    # since expansion is a recall booster, not something retrieval requires
    # to function (rewritten_query alone is always searched too).
    expanded_queries: list[str]
    dense_results: list[tuple[Document, float]]
    bm25_results: list[tuple[Document, float]]
    fused_results: list[tuple[Document, float]]
    reranked_results: list[tuple[Document, float]]
    # One row per chunk in reranked_results, carrying its 1-indexed rank in
    # each earlier stage (dense/BM25/fused) alongside its final position —
    # built by rerank_node from lists the graph already computed, purely so
    # the UI's retrieval-proof table (see UI_UPGRADE_SPEC.md) has something
    # honest to render instead of asserting "hybrid retrieval happened."
    retrieval_proof: list[dict]
    compressed_docs: list[Document]
    chat_history: list[dict[str, str]]
    answer: str
    # Which model/provider actually produced `answer` — set by generate_node
    # from get_completion_with_model's return, not a fixed setting value,
    # since automatic Groq-model and Groq->Ollama fallback (app/llm/
    # llm_gateway.py) both mean the answering model isn't predictable from
    # config alone. Exists so "which model answered" is checkable on the
    # record instead of assumed, the same reasoning retrieval_proof above
    # was built on.
    answer_model: str
    answer_provider: str


if __name__ == "__main__":
    # Tiny self-test: TypedDict with total=False means a partial dict (just
    # the caller-supplied keys) is a valid GraphState — that's the whole
    # point, so prove it type-checks and behaves like a plain dict at runtime.
    initial_state: GraphState = {"conversation_id": "demo", "query": "What is RRF?"}
    print(f"Initial state: {initial_state}")
    assert "rewritten_query" not in initial_state
    print("OK: partial state is valid, missing keys are simply absent.")
