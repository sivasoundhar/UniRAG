"""
Central configuration for UniRAG.

Every tunable that shows up more than once in the codebase (chunk sizes, k
values, model names, paths) lives here instead of being hard-coded at the
call site. This is what "no magic numbers" means in practice: the answer to
"why 500 tokens per chunk?" should be a comment on one line in this file,
not archaeology across five modules.
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
    # is down or the key is missing. See app/llm/llm_gateway.py.
    llm_provider: str = "groq"
    groq_api_key: str = ""

    # --- Vector store ------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma_db"

    # --- OCR ---------------------------------------------------------------
    tesseract_cmd: str = "/usr/bin/tesseract"

    # --- Vision (image understanding, app/loaders/ocr_loader.py) -----------
    # OCR only extracts *text* — a photo/scan with no printed text in it
    # (an X-ray, a diagram, a CAE render) OCRs to nothing, which the loader
    # used to treat as a hard failure. This model lets it fall back to
    # *describing* the image instead, so image uploads with no text still
    # produce something indexable. Groq-only, no Ollama vision fallback:
    # unlike the text gateway, this project isn't running a local vision
    # model (not installed, and irrelevant for a hosted demo deployment
    # that won't have Ollama available at all), so if this fails the
    # upload fails loudly rather than silently degrading to a much
    # smaller/absent local option.
    #
    # Verified for real against client.models.list() on a live Groq
    # connection — every earlier guess here (from Groq's own docs, which
    # gave self-contradictory answers across repeated fetches, likely
    # JS-rendered content the fetcher couldn't read reliably) turned out
    # wrong: "meta-llama/llama-4-scout-17b-16e-instruct" and
    # "...-maverick-..." don't exist in the real model list at all.
    # qwen/qwen3.6-27b is the only vision-capable model this account
    # actually has access to right now — no fallback model exists to list.
    # Note: on Windows machines with Avast installed, its HTTPS-scanning
    # feature can intercept the connection with its own certificate
    # (trusted by Windows, not by Python's cert store), which looks like a
    # network/TLS failure but isn't a Groq-side or code-side problem.
    groq_vision_model: str = "qwen/qwen3.6-27b"
    groq_vision_fallback_models: list[str] = []

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

    # --- Vector store / retrieval -------------------------------------------
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

    # --- LLM-backed query tools ----------------------------------------------
    # Set to the 70B model as primary — better answer quality is worth the
    # extra latency. Worth knowing: this same setting
    # (and its fallback list) drives every Groq text call in the app, not
    # just chat answers — rewrite_node and expand_node (query rewrite/
    # expansion) use it too, so they get slower and more expensive per
    # turn as a side effect, not just generate_node. The original 8B
    # default was chosen specifically because rewrite/expansion are "short
    # transformation tasks, not reasoning tasks" where the smallest
    # reliable model was the right cost/latency fit — that reasoning still
    # holds, it's just been deliberately overridden here in favor of
    # better final-answer quality everywhere. If rewrite/expand latency
    # becomes a problem, the real fix is splitting this into two settings
    # (a fast one for query tools, a bigger one for generate_node), not
    # reverting this.
    groq_model: str = "llama-3.3-70b-versatile"
    # If groq_model errors (rate-limited, decommissioned, transient Groq-side
    # fault), llm_gateway tries these next, in order, before giving up on
    # Groq entirely. All four are Groq's current free/production-tier text
    # models (checked against console.groq.com/docs/models) — deliberately
    # excludes whisper-* (audio, not chat) and groq/compound* (agentic
    # systems with built-in tools, a different call shape than a plain
    # completion).
    groq_fallback_models: list[str] = [
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
    ]
    # Local Ollama fallback model — separate from groq_model since the two
    # providers don't share a model namespace (Groq's model IDs aren't
    # valid Ollama tags and vice versa).
    ollama_model: str = "llama3.2"
    # Enough alternate phrasings to catch synonyms/related terms without
    # diluting retrieval with too many near-duplicate queries.
    query_expansion_n: int = 3

    # --- Rerank --------------------------------------------------------------
    # Small (~80MB), CPU-friendly, open, purpose-built for query-passage
    # relevance scoring — unlike the bi-encoder used for dense retrieval, a
    # cross-encoder reads the query and passage together, which is more
    # accurate but too slow to run over the whole corpus.
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Final context size handed downstream — the cross-encoder's job is to
    # cut retrieval's top_k down hard to just the passages worth paying for.
    rerank_top_n: int = 3

    # --- Compression ----------------------------------------------------------
    # How many of each reranked chunk's sentences survive compression — enough
    # to keep the answer-bearing sentence(s) without handing the LLM the full
    # chunk text, which is most valuable once rerank_top_n chunks are already
    # each carrying some irrelevant surrounding sentences.
    compression_sentences_per_doc: int = 3

    # --- Memory ------------------------------------------------------------------
    # How many past (user, assistant) turn-pairs get fed back into rewrite and
    # generate each turn. Bounds prompt growth over a long conversation instead
    # of replaying the entire history every time.
    conversation_memory_max_turns: int = 5

    # --- API -----------------------------------------------------------------------
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
