"""
One function per pipeline step. See build_graph.py for how these wire
together into a graph, the ASCII diagram, and the "why this works" note.

Each node takes the full GraphState and returns only the keys it changed —
LangGraph merges that partial dict back into state (default: last write
wins), so a node never needs to know about fields it doesn't touch.
"""

from app.compression.context_compressor import compress_context
from app.config.settings import settings
from app.graph.state import GraphState
from app.llm.llm_gateway import get_completion
from app.memory.conversation_memory import append_turn, get_history
from app.query.rewrite import rewrite_query
from app.rerank.cross_encoder_rerank import rerank
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore

_vector_store: VectorStore | None = None
_bm25_retriever: BM25Retriever | None = None


def _get_vector_store() -> VectorStore:
    """Lazily open the Chroma collection once per process, not once per turn."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_bm25_retriever() -> BM25Retriever:
    """
    Lazily build the BM25 index once per process from whatever's in Chroma
    at that moment. A real deployment would rebuild this on ingest, not once
    ever — fine for a single-run demo, a known limitation past that.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
        _bm25_retriever.build(_get_vector_store().get_all_documents())
    return _bm25_retriever


def rewrite_node(state: GraphState) -> dict:
    """
    Resolve the raw query into a retrieval-friendly one. On the first turn
    of a conversation there's no history, so the query goes through as-is;
    on later turns, prior turns are prepended so rewrite_query's existing
    "resolve vague references if the query provides enough context" prompt
    (see app/query/rewrite.py) has something to resolve "it"/"that" against
    — without needing to change that module at all.
    """
    history = get_history(state["conversation_id"])

    if not history:
        rewritten = rewrite_query(state["query"])
    else:
        history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        augmented_input = f"Conversation so far:\n{history_text}\n\nNew question: {state['query']}"
        rewritten = rewrite_query(augmented_input)

    return {"rewritten_query": rewritten, "chat_history": history}


def retrieve_node(state: GraphState) -> dict:
    """Dense and BM25 search on the rewritten query, ahead of fusion."""
    query = state["rewritten_query"]
    dense_results = _get_vector_store().similarity_search(query, k=settings.retrieval_top_k)
    bm25_results = _get_bm25_retriever().search(query, k=settings.retrieval_top_k)
    return {"dense_results": dense_results, "bm25_results": bm25_results}


def fuse_node(state: GraphState) -> dict:
    """Merge dense and BM25 rankings with RRF."""
    fused_results = reciprocal_rank_fusion([state["dense_results"], state["bm25_results"]])
    return {"fused_results": fused_results}


def rerank_node(state: GraphState) -> dict:
    """Re-score the fused candidates with the cross-encoder for final ordering."""
    documents = [doc for doc, _score in state["fused_results"]]
    reranked_results = rerank(state["rewritten_query"], documents, top_n=settings.rerank_top_n)
    return {"reranked_results": reranked_results}


def compress_node(state: GraphState) -> dict:
    """Trim each reranked chunk down to its most query-relevant sentences."""
    documents = [doc for doc, _score in state["reranked_results"]]
    compressed_docs = compress_context(state["rewritten_query"], documents)
    return {"compressed_docs": compressed_docs}


def generate_node(state: GraphState) -> dict:
    """
    Build one prompt from chat history + compressed context + the
    *original* question (not the rewritten one — the rewrite was for
    retrieval; the user should be answered in their own words) and call
    the LLM gateway for a final answer.
    """
    context_text = "\n\n".join(doc.page_content for doc in state["compressed_docs"])
    history = state.get("chat_history", [])
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    system_prompt = (
        "You are a retrieval-augmented assistant. Answer ONLY using the "
        "context below. Do not use your own general knowledge or training "
        "data to answer — this includes writing code, solving problems, or "
        "answering general-knowledge questions unrelated to the context, "
        "even if you know the answer and even if the context is silent on "
        "it. If the context does not contain the answer, respond with "
        "exactly this sentence and nothing else: \"I don't have information "
        "about that in the indexed documents.\" Do not add a caveat and "
        "then answer anyway. Keep answers concise."
    )
    prompt_parts = []
    if history_text:
        prompt_parts.append(f"Conversation so far:\n{history_text}")
    prompt_parts.append(f"Context:\n{context_text}")
    prompt_parts.append(f"Question: {state['query']}")

    answer = get_completion("\n\n".join(prompt_parts), system=system_prompt)
    return {"answer": answer}


def save_turn_node(state: GraphState) -> dict:
    """Record this turn so the next one has it as history. No state change."""
    append_turn(state["conversation_id"], "user", state["query"])
    append_turn(state["conversation_id"], "assistant", state["answer"])
    return {}


if __name__ == "__main__":
    # Tiny self-test: fuse_node is the one node here with no model/LLM/DB
    # dependency, so it's the one that can run in isolation. The rest
    # (rewrite/retrieve/rerank/compress/generate/save_turn) are exercised
    # end-to-end, with real data, by build_graph.py's self-test instead —
    # testing them individually here would just duplicate that run.
    from langchain_core.documents import Document

    doc_a = Document(page_content="RRF fuses BM25 and dense retrieval.", metadata={"chunk_id": "a"})
    doc_b = Document(page_content="Tesseract reads text out of images.", metadata={"chunk_id": "b"})

    fake_state: GraphState = {
        "dense_results": [(doc_a, 0.9), (doc_b, 0.5)],
        "bm25_results": [(doc_b, 3.0), (doc_a, 1.0)],
    }
    result = fuse_node(fake_state)
    print("Fused results:")
    for doc, score in result["fused_results"]:
        print(f"  score={score:.4f} chunk_id={doc.metadata['chunk_id']!r}")
    assert "fused_results" in result and len(result["fused_results"]) == 2
    print("OK: fuse_node merges both ranked lists into one.")
