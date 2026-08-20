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
from app.llm.llm_gateway import get_completion, get_completion_with_model
from app.logging_config.logger import get_logger
from app.memory.conversation_memory import append_turn, get_history
from app.query.expansion import expand_query
from app.query.rewrite import rewrite_query
from app.rerank.cross_encoder_rerank import rerank
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore

_logger = get_logger(__name__)
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


def expand_node(state: GraphState) -> dict:
    """
    Generate settings.query_expansion_n alternate phrasings of the
    rewritten query (app/query/expansion.py — written earlier, only wired
    into the graph here). A separate node rather than folded into retrieve_node, matching
    this file's "one node per pipeline step" pattern (see its module
    docstring) and so the ASCII diagram in build_graph.py has a real box
    for it, not a hidden side effect of the retrieve step.

    Best-effort, not required: falls back to no variants (retrieval still
    runs on rewritten_query alone, exactly like before this node existed)
    if the expansion LLM call itself fails, instead of failing the whole
    turn over a recall *booster*. This matters concretely whenever Groq
    connectivity is intermittent — rewrite_node's own LLM call has no such fallback
    because retrieval has no fallback for a missing rewritten_query, but
    retrieval works fine on rewritten_query alone with zero expansions.
    """
    try:
        variants = expand_query(state["rewritten_query"])
    except Exception as e:
        _logger.warning("Query expansion failed (%s) — continuing with rewritten_query only", e)
        variants = []
    return {"expanded_queries": variants}


def retrieve_node(state: GraphState) -> dict:
    """
    Dense and BM25 search across the rewritten query plus every expanded
    variant, ahead of fusion. Each retriever's per-variant results are
    RRF-merged into that retriever's single combined list before returning
    — reusing reciprocal_rank_fusion for this second purpose (collapsing N
    phrasings of the same retriever into one ranking, not just merging
    dense vs BM25) because the mechanism is identical either way: it only
    ever cares about each list's rank *positions*. This keeps
    dense_results/bm25_results the same shape they've always been, so
    fuse_node, rerank_node, and the retrieval-proof table (which reads
    "the" dense/BM25 rank per chunk) need zero changes to benefit from
    expansion.
    """
    all_queries = [state["rewritten_query"], *state.get("expanded_queries", [])]

    dense_per_query = [
        _get_vector_store().similarity_search(q, k=settings.retrieval_top_k) for q in all_queries
    ]
    bm25_per_query = [_get_bm25_retriever().search(q, k=settings.retrieval_top_k) for q in all_queries]

    # Skip the RRF call entirely for the (single-query, no variants) case —
    # fusing a lone list with itself is a no-op, but a needless one.
    dense_results = dense_per_query[0] if len(dense_per_query) == 1 else reciprocal_rank_fusion(dense_per_query)
    bm25_results = bm25_per_query[0] if len(bm25_per_query) == 1 else reciprocal_rank_fusion(bm25_per_query)

    return {"dense_results": dense_results, "bm25_results": bm25_results}


def fuse_node(state: GraphState) -> dict:
    """Merge dense and BM25 rankings with RRF."""
    fused_results = reciprocal_rank_fusion([state["dense_results"], state["bm25_results"]])
    return {"fused_results": fused_results}


def _rank_map(ranked_list: list[tuple]) -> dict:
    """chunk_id -> this one ranker's 1-indexed rank position, for O(1) lookup below."""
    return {doc.metadata["chunk_id"]: rank for rank, (doc, _score) in enumerate(ranked_list, start=1)}


def _build_retrieval_proof(
    dense_results: list[tuple],
    bm25_results: list[tuple],
    fused_results: list[tuple],
    reranked_results: list[tuple],
) -> list[dict]:
    """
    Takes: the four ranked lists this graph run already produced — no new
    retrieval work, just reading positions that already exist in each list.
    Returns: one row per chunk that survived to the final reranked set,
    with its 1-indexed rank at each earlier stage (None if that ranker's
    list didn't surface it at all — e.g. BM25 missed a chunk dense
    retrieval found, or vice versa).
    Use this: to make "hybrid retrieval + RRF + reranking are real, not
    just claimed" checkable on the record instead of asserted — the
    Streamlit UI's retrieval-proof table renders this list directly under
    each answer.
    """
    dense_ranks = _rank_map(dense_results)
    bm25_ranks = _rank_map(bm25_results)
    fused_ranks = _rank_map(fused_results)

    proof = []
    for position, (doc, _score) in enumerate(reranked_results, start=1):
        chunk_id = doc.metadata["chunk_id"]
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:60].strip().replace("\n", " ")
        ellipsis = "…" if len(doc.page_content) > 60 else ""
        proof.append(
            {
                "chunk": f"{source} — {preview}{ellipsis}",
                "bm25_rank": bm25_ranks.get(chunk_id),
                "dense_rank": dense_ranks.get(chunk_id),
                "fused_rank": fused_ranks.get(chunk_id),
                "rerank_position": position,
            }
        )
    return proof


def rerank_node(state: GraphState) -> dict:
    """
    Re-score the fused candidates with the cross-encoder for final ordering,
    and build the retrieval-proof rows (see _build_retrieval_proof) from the
    dense/BM25/fused/reranked lists this turn produced — done here, not as a
    separate node, because this is the one point in the graph where all four
    lists exist in state at once.
    """
    documents = [doc for doc, _score in state["fused_results"]]
    reranked_results = rerank(state["rewritten_query"], documents, top_n=settings.rerank_top_n)
    retrieval_proof = _build_retrieval_proof(
        dense_results=state["dense_results"],
        bm25_results=state["bm25_results"],
        fused_results=state["fused_results"],
        reranked_results=reranked_results,
    )
    return {"reranked_results": reranked_results, "retrieval_proof": retrieval_proof}


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
    # Each chunk is labeled with its own source, not just concatenated raw
    # — without this, the model has no way to tell where one retrieved
    # document's text ends and the next begins, and *will* blend them into
    # one answer as if they were continuous. Caught this for real: two
    # unrelated images (bird.jpg, an animated-show still with no bird in
    # it; car.jpg, an actual bird illustration) got retrieved for the same
    # question, and with no boundary between their concatenated
    # page_content, the final answer merged car.jpg's visual details
    # ("gradient background...") into a description it presented as
    # bird.jpg's — plus, seeing "bird" nowhere disambiguated from
    # "car.jpg" content that genuinely was about a bird, the model
    # defaulted to inventing "the subject, a bird" for bird.jpg too, which
    # its own actual content never supports. This isn't image-specific —
    # any two chunks from different source documents were equally at risk
    # of being blended; images just made the wrong-answer visible.
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in state["compressed_docs"]
    )
    history = state.get("chat_history", [])
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    # The off-topic refusal is still a fixed, verbatim sentence, not
    # something the model composes freely — that's deliberate, not an
    # oversight. It guards against an observed real failure: a smaller
    # local model (llama3.2:3b via Ollama) would hedge
    # ("Context doesn't say, however...") and then answer anyway from its
    # own training data, defeating the entire point of "answers grounded
    # in your own documents." A free-form refusal reopens that exact door
    # — nothing stops the model from wrapping a hedge in politer words. So
    # the wording itself does the work: it's warm and redirects the user
    # back to what the assistant *can* do, without ever giving the model
    # room to improvise past "no."
    #
    # Greetings/small talk are a deliberate, narrow carve-out from that
    # same rule, added after observing a plain "hi" get the identical
    # flat refusal as a real off-topic question ("write me a function"),
    # which read as broken/robotic rather than careful. The boundary is
    # drawn as tight as the original anti-hedging guard needs it to stay:
    # ONLY a message that is pure greeting/small talk with no real
    # question in it gets a warm reply instead of the fixed sentence.
    # Anything with actual content — a real question, a request, a topic —
    # still goes through the exact same strict "answer only from context,
    # or refuse verbatim" rule as before, so this can't become a second
    # loophole for the model to hedge-then-answer through.
    system_prompt = (
        "You are a friendly, helpful retrieval-augmented assistant — keep a "
        "warm, polite tone in every answer, including when you can't help.\n\n"
        "If — and only if — the message is PURELY a greeting or small talk "
        "with no real question or request in it (e.g. \"hi\", \"hello\", "
        "\"good morning\", \"how are you\", \"thanks\"), reply warmly and "
        "briefly, and invite them to ask something about the indexed "
        "documents. Do not use the fixed refusal sentence below for this "
        "case, and do not describe or summarize the documents unprompted — "
        "just a short, friendly greeting back plus an invitation to ask.\n\n"
        "For every other message: Answer ONLY using the context below. Do "
        "not use your own general knowledge or training data to answer — "
        "this includes writing code, solving problems, or answering "
        "general-knowledge questions unrelated to the context — even if "
        "you know the answer, even if the context is silent on it. If the "
        "context does not contain the answer to what's being asked — "
        "including when the message is off-topic or unrelated to what's "
        "been indexed — respond with exactly this sentence and nothing "
        "else, do not vary the wording: \"That's not something I can find "
        "in the documents I have indexed — I'm best used for questions "
        "about that material, so feel free to ask me something from "
        "there!\" Do not add a caveat and then answer anyway. Keep answers "
        "concise. The context is divided into blocks marked [Source: "
        "<filename>] — treat each block as describing only its own "
        "source, never blend, merge, or attribute details from one "
        "source's block onto another, even if their filenames or subjects "
        "seem related. If the question asks about a specific named "
        "source and multiple sources were retrieved, only use the block(s) "
        "whose [Source: ...] name actually matches."
    )
    prompt_parts = []
    if history_text:
        prompt_parts.append(f"Conversation so far:\n{history_text}")
    prompt_parts.append(f"Context:\n{context_text}")
    prompt_parts.append(f"Question: {state['query']}")

    completion = get_completion_with_model("\n\n".join(prompt_parts), system=system_prompt)
    return {"answer": completion.text, "answer_model": completion.model, "answer_provider": completion.provider}


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

    # _build_retrieval_proof is also DB/model-free (it only reads positions
    # out of lists that already exist), so it can run standalone here too —
    # doc_b ranks #1 by BM25 but #2 by dense; the proof rows should show
    # exactly that disagreement, not paper over it.
    proof = _build_retrieval_proof(
        dense_results=fake_state["dense_results"],
        bm25_results=fake_state["bm25_results"],
        fused_results=result["fused_results"],
        reranked_results=[(doc_b, 4.1), (doc_a, 2.7)],
    )
    print("\nRetrieval proof:")
    for row in proof:
        print(f"  {row}")
    assert proof[0]["rerank_position"] == 1 and proof[0]["bm25_rank"] == 1 and proof[0]["dense_rank"] == 2
    print("OK: retrieval proof rows carry each chunk's real rank at every stage.")
