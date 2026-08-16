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
    dense_results: list[tuple[Document, float]]
    bm25_results: list[tuple[Document, float]]
    fused_results: list[tuple[Document, float]]
    reranked_results: list[tuple[Document, float]]
    compressed_docs: list[Document]
    chat_history: list[dict[str, str]]
    answer: str


if __name__ == "__main__":
    # Tiny self-test: TypedDict with total=False means a partial dict (just
    # the caller-supplied keys) is a valid GraphState — that's the whole
    # point, so prove it type-checks and behaves like a plain dict at runtime.
    initial_state: GraphState = {"conversation_id": "demo", "query": "What is RRF?"}
    print(f"Initial state: {initial_state}")
    assert "rewritten_query" not in initial_state
    print("OK: partial state is valid, missing keys are simply absent.")
