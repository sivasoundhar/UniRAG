"""
Central configuration for UniRAG.

Every tunable that shows up more than once in the codebase (chunk sizes, k
values, model names, paths) lives here instead of being hard-coded at the
call site. This is what "no magic numbers" means in practice: if an
interviewer asks "why 500 tokens per chunk?", the answer should be a comment
on one line in this file, not archaeology across five modules.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    App-wide settings loaded from environment variables / .env.

    Takes: nothing (Pydantic Settings reads from the environment at
    instantiation time).
    Returns: a validated Settings object.
    Use this: import the `settings` singleton at the bottom of this module
    anywhere a module needs a config value — never read os.environ directly.
    """

    # --- LLM gateway -----------------------------------------------------
    # Groq is free-tier and fast; Ollama is the local fallback when the API
    # is down or the key is missing. See app/llm/llm_gateway.py (Day 6).
    llm_provider: str = "groq"
    groq_api_key: str = ""

    # --- Vector store ------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma_db"

    # --- OCR ---------------------------------------------------------------
    tesseract_cmd: str = "/usr/bin/tesseract"

    # --- Embeddings ----------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Chunking ------------------------------------------------------------
    # 500 chars keeps a chunk roughly paragraph-sized for BGE-small (512
    # token context) without truncation risk once special tokens are added.
    chunk_size: int = 500
    # 15% overlap is enough to keep a sentence that straddles a chunk
    # boundary readable in at least one chunk, without bloating index size.
    chunk_overlap: int = 75
    # Semantic chunker splits where cosine distance between consecutive
    # sentence embeddings jumps above this percentile of all jumps in the
    # document — i.e. "this sentence is topically further from its neighbor
    # than most sentence-pairs are."
    semantic_breakpoint_percentile: int = 95

    # --- Vector store / retrieval (Day 2) -----------------------------------
    chroma_collection_name: str = "unirag_chunks"
    # BGE models are trained asymmetrically: the query gets this instruction
    # prefix, indexed passages never do. Skipping it on the query side still
    # "works" but measurably hurts recall — this is a BGE-specific quirk, not
    # a general embedding-model rule.
    bge_query_instruction: str = "Represent this sentence for searching relevant passages: "
    # How many candidates each retriever (dense, BM25) surfaces before fusion.
    # Wider than the final rerank_top_n so RRF has enough signal to work with.
    retrieval_top_k: int = 5
    # RRF's smoothing constant, taken from the original paper (Cormack et al.
    # 2009). A small k would let rank 1 dominate the fused score; 60 keeps
    # the top handful of ranks from any single retriever roughly comparable.
    rrf_k: int = 60

    # --- LLM-backed query tools (Day 3) -------------------------------------
    # Free-tier, fast, instruction-following model — rewrite/expansion are
    # short transformation tasks, not reasoning tasks, so the smallest Groq
    # model that follows instructions reliably is the right cost/latency fit.
    groq_model: str = "llama-3.1-8b-instant"
    # Local Ollama fallback model — separate from groq_model since the two
    # providers don't share a model namespace (Groq's model IDs aren't
    # valid Ollama tags and vice versa).
    ollama_model: str = "llama3.2"
    # Enough alternate phrasings to catch synonyms/related terms without
    # diluting retrieval with too many near-duplicate queries.
    query_expansion_n: int = 3

    # --- Rerank (Day 3) ------------------------------------------------------
    # Small (~80MB), CPU-friendly, open, purpose-built for query-passage
    # relevance scoring — unlike the bi-encoder used for dense retrieval, a
    # cross-encoder reads the query and passage together, which is more
    # accurate but too slow to run over the whole corpus.
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Final context size handed downstream — the cross-encoder's job is to
    # cut retrieval's top_k down hard to just the passages worth paying for.
    rerank_top_n: int = 3

    # --- Compression (Day 4) -------------------------------------------------
    # How many of each reranked chunk's sentences survive compression — enough
    # to keep the answer-bearing sentence(s) without handing the LLM the full
    # chunk text, which is most valuable once rerank_top_n chunks are already
    # each carrying some irrelevant surrounding sentences.
    compression_sentences_per_doc: int = 3

    # --- Memory (Day 4) --------------------------------------------------------
    # How many past (user, assistant) turn-pairs get fed back into rewrite and
    # generate each turn. Bounds prompt growth over a long conversation instead
    # of replaying the entire history every time.
    conversation_memory_max_turns: int = 5

    # --- API (Day 6) -----------------------------------------------------------
    # Where POST /api/v1/upload writes incoming files before loading/chunking
    # them — kept out of chroma_persist_dir since one is source files, the
    # other is the derived index; gitignored, same as chroma_persist_dir.
    upload_dir: str = "./data/uploads"

    # --- App -----------------------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Import this, don't instantiate Settings() yourself — keeps config
# construction (and .env parsing) happening exactly once per process.
settings = Settings()


if __name__ == "__main__":
    # Tiny self-test: prove .env loads and defaults are sane.
    print("Loaded settings:")
    for field_name in Settings.model_fields:
        print(f"  {field_name} = {getattr(settings, field_name)!r}")
