"""
Wire app/graph/nodes.py into a compiled LangGraph pipeline.

    START
      │
      ▼
 ┌──────────┐
 │ rewrite  │   resolve the query using chat history (pronouns, typos)
 └────┬─────┘
      ▼
 ┌──────────┐
 │ retrieve │   dense (Chroma) + BM25 search on the rewritten query
 └────┬─────┘
      ▼
 ┌──────────┐
 │   fuse   │   RRF merges the two ranked lists
 └────┬─────┘
      ▼
 ┌──────────┐
 │  rerank  │   cross-encoder re-scores the fused top-k
 └────┬─────┘
      ▼
 ┌──────────┐
 │ compress │   trim each chunk to its most relevant sentences
 └────┬─────┘
      ▼
 ┌──────────┐
 │ generate │   LLM answers using history + compressed context
 └────┬─────┘
      ▼
 ┌───────────┐
 │ save_turn │   record this turn for the next one's history
 └─────┬─────┘
       ▼
      END

Why this works: it's a straight line, not a branching graph, because
nothing here needs to be decided conditionally yet — every turn runs every
step in the same order. LangGraph earns its keep once a step becomes
conditional (e.g. "skip retrieval for a pure follow-up question" or "route
to a different generation prompt for a summarization request") — that's a
natural extension point past Day 4, not something today's finish line
("one multi-turn conversation works end-to-end, memory carries context")
needs. Building branching logic before there's a real branching decision to
make would be solving a problem this app doesn't have yet.

The state (GraphState) is what makes this a graph instead of just a
function that calls seven other functions in sequence: each node only
declares the keys it reads and writes, so the pipeline can be inspected,
re-ordered, or extended one node at a time without every node needing to
know about every other node's internals.
"""

import time

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    compress_node,
    fuse_node,
    generate_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    save_turn_node,
)
from app.graph.state import GraphState
from app.logging_config.logger import get_logger, log_request

_logger = get_logger(__name__)


def build_graph():
    """
    Takes: nothing.
    Returns: a compiled LangGraph runnable — call .invoke(state) on it with
    at least {"conversation_id": ..., "query": ...} to run one turn.
    Use this: once per process to get the compiled graph, then invoke it
    once per user turn.
    """
    graph = StateGraph(GraphState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save_turn", save_turn_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "save_turn")
    graph.add_edge("save_turn", END)

    return graph.compile()


def run_turn(graph, conversation_id: str, query: str) -> GraphState:
    """
    Takes: a compiled graph (from build_graph()), a conversation_id, and
    the user's query.
    Returns: the final GraphState for this turn — same as calling
    graph.invoke() directly.
    Use this: instead of calling graph.invoke() directly, whenever the
    Day 5 finish line's per-request logging (latency, retrieval count,
    sources) needs to actually happen — this is the one seam every request
    passes through, so it's the right place to time and log, rather than
    threading logging calls into individual nodes.
    """
    start = time.perf_counter()
    result = graph.invoke({"conversation_id": conversation_id, "query": query})
    latency_ms = (time.perf_counter() - start) * 1000

    log_request(
        _logger,
        query=query,
        latency_ms=latency_ms,
        retrieval_count=len(result.get("reranked_results", [])),
        sources=[doc.metadata.get("source", "unknown") for doc in result.get("compressed_docs", [])],
    )
    return result


if __name__ == "__main__":
    # Day 4 finish line: seed a couple of documents about RRF into the
    # vector store, then run two turns of one conversation. Turn 2 uses
    # "it" to refer to turn 1's topic — if memory carries context, rewrite
    # should resolve "it" into something retrieval can actually search for,
    # and the answer should stay coherent with turn 1.
    from langchain_core.documents import Document

    from app.retrieval.vector_store import VectorStore

    VectorStore().add_documents([
        Document(page_content="Reciprocal rank fusion (RRF) merges a dense retriever's ranked list and a BM25 retriever's ranked list into one combined ranking.", metadata={"source": "rrf_intro.txt"}),
        Document(page_content="RRF uses each document's rank position, not its raw score, because BM25 and cosine similarity scores live on unrelated scales.", metadata={"source": "rrf_why_rank.txt"}),
    ])

    app_graph = build_graph()
    conversation_id = "demo-conversation"

    turn_1 = run_turn(app_graph, conversation_id, "What is reciprocal rank fusion?")
    print(f"Turn 1 query:     {turn_1['query']!r}")
    print(f"Turn 1 rewritten: {turn_1['rewritten_query']!r}")
    print(f"Turn 1 answer:    {turn_1['answer']!r}\n")

    turn_2 = run_turn(app_graph, conversation_id, "Why does it use rank instead of raw score?")
    print(f"Turn 2 query:     {turn_2['query']!r}")
    print(f"Turn 2 rewritten: {turn_2['rewritten_query']!r}")
    print(f"Turn 2 answer:    {turn_2['answer']!r}")

    assert "rrf" in turn_2["rewritten_query"].lower() or "reciprocal rank fusion" in turn_2["rewritten_query"].lower(), (
        "turn 2's rewrite should have resolved 'it' using turn 1's history"
    )
    print("\nOK: turn 2's pronoun resolved using turn 1's history — memory carried context.")
