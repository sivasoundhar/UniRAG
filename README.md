# UniRAG

A from-scratch, production-shaped Retrieval-Augmented Generation platform —
built as a 6-day portfolio project, not a hosted product. UniRAG is designed
as a reusable **AI Knowledge Core**: one hybrid-retrieval RAG API meant to
sit underneath multiple future downstream copilots (an Engineering AI
Copilot, a Medical AI Copilot — both hypothetical future consumers, not
built yet), rather than a single-purpose chatbot.

This project exists to be **read**, not just run. Every non-obvious design
choice — why RRF instead of raw-score fusion, why regex guardrails instead
of an LLM self-check, why the UI never imports the backend — has an inline
"why this works" comment in the code, specifically so it can be explained
out loud in an interview. This README is the map; the code and
[`docs/CLAUDE.md`](./docs/CLAUDE.md) are the territory.

```
                AI Knowledge Core (UniRAG)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  Engineering AI Copilot       Medical AI Copilot
     (future, separate)         (future, separate)
```

No domain-specific logic lives in `app/` — every module is importable and
usable standalone. A downstream copilot would call this API and add its own
domain framing on top, not fork the core. (Architecturally sound, not yet
built — and today there's no per-domain data isolation either; see
[Known limitations](#known-limitations).)

> **Live demo:** not yet deployed — `render.yaml` is prepared but unverified
> against a live account (see [Deployment](#deployment)).
> **Screenshots:** TODO before publishing — add 2-3 screenshots of the Chat
> page (pipeline panel + retrieval-proof table). A static HTML mockup of the
> intended look exists at `unirag_ui_v3_demo.html` as a reference in the
> meantime.

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Key design decisions](#key-design-decisions)
- [Tech stack](#tech-stack)
- [API reference](#api-reference)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Build process & interview talking points](#build-process--interview-talking-points)
- [Known limitations](#known-limitations)
- [License](#license)
- [Further reading](#further-reading)

## Features

UniRAG's UI is built to make the backend's mechanics visible to a stranger,
fast — not to look like a generic chat-with-your-docs SaaS demo.

- **Retrieval-proof table.** Every answer is followed by a table showing,
  per retrieved chunk, its BM25 rank, dense rank, fused (RRF) rank, and
  final reranked position — with highlighting when a chunk was corroborated
  by more than one signal. This is the strongest evidence in the whole UI
  that hybrid retrieval, RRF, and reranking are real mechanics, not
  marketing copy.
- **Always-visible pipeline panel.** A "how this works" strip
  (rewrite → expand → hybrid retrieve → rerank → compress → guardrails)
  sits on the Chat page at all times, with monochrome glyph icons, and
  pulses across every stage while a `/chat` call is in flight — deliberately
  *not* a fake sequential per-stage animation, since `/chat` is one blocking
  call and doesn't stream; animating stage-by-stage would misrepresent that.
- **Zero-setup sample corpus.** A 3-document sample corpus (about hybrid
  retrieval, RRF/reranking, and guardrails — UniRAG explaining its own
  pipeline) is seeded automatically at API startup, idempotently, so Chat
  and Search have something real to query the moment the app is up.
- **Document lifecycle controls.** Per document: 🙈 hide (soft delete, kept
  fully restorable), 🗑️ delete (permanent), and ↩️ restore from a "Hidden
  documents" expander — backed by a real `active` flag in the vector store,
  not a UI-only toggle.
- **Search page (retrieval-only).** Runs the hybrid retrieval + rerank stack
  with no LLM call at all, showing raw source/score/text per result —
  useful for inspecting retrieval quality in isolation from generation
  quality.
- **Live meta-row per answer.** Latency, retrieval count, reranked count,
  and which model/provider actually answered (`answer_model` /
  `answer_provider` — meaningful given Groq can silently fail over to other
  Groq models, or to Ollama, mid-session).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Streamlit UI  (streamlit_app/)                                       │
│ thin HTTP client — never imports app/, talks only to the REST API    │
└──────────────────────────────────────────────────────────────────────┘
                                   │  HTTP (requests library)
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FastAPI  (app/main.py) — every route is a plain sync `def`           │
│ upload · documents (delete/restore) · chat · search · health · stats │
│ (global exception handler guarantees every response is JSON)         │
└──────────────────────────────────────────────────────────────────────┘
                                   │  /api/v1/chat only
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ input_guard  (app/guardrails/) — NeMo Guardrails, regex rail only    │
│ PII patterns + jailbreak patterns on the raw query → 400 if blocked  │
└──────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LangGraph pipeline  (app/graph/build_graph.py) — 8 nodes, linear     │
│                                                                       │
│ rewrite → expand → retrieve → fuse → rerank → compress               │
│                                     → generate → save_turn           │
└──────────────────────────────────────────────────────────────────────┘
                                   │
             ┌─────────────────────┬─────────────┐
             ▼                                   ▼
┌─────────────────────────┐    ┌────────────────────────────────────┐
│ Retrieval core          │    │ LLM gateway                        │
│  Chroma (dense, cosine) │    │  Groq (primary + fallback)         │
│  BM25Retriever          │    │  → Ollama (local)                  │
│  reciprocal_rank_fusion │    │  used by rewrite/expand/generate   │
│  cross-encoder rerank   │    │  vision model for OCR-empty images │
└─────────────────────────┘    └────────────────────────────────────┘
             │                                   │
             └─────────────────────┴─────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ output_guard  (app/guardrails/) — regex PII-leak check on the        │
│ generated answer → 400 if blocked                                    │
└──────────────────────────────────────────────────────────────────────┘
                                   ▼
                         back through FastAPI to the client
```

Guardrails deliberately live **outside** the LangGraph pipeline, wired at
the FastAPI route layer — see [Key design decisions](#key-design-decisions).

## How it works

A `/api/v1/chat` request passes through 10 steps end-to-end: two guardrail
checks at the API layer, wrapping the 8-node LangGraph pipeline
(`app/graph/build_graph.py`) — a straight line, no branching, every time.

1. **input_guard** *(API layer, not in the graph)* — regex rail checks the
   raw query against PII patterns (email/SSN/credit-card/phone-shaped) and
   known jailbreak framings. Blocks with a 400 before anything else runs.
2. **rewrite** — resolves the raw query into a standalone question, using
   chat history from turn 2 onward (so "what about the second one?"
   resolves against the prior turn) via `app/query/rewrite.py`.
3. **expand** — generates `query_expansion_n` alternate phrasings of the
   rewritten query via `app/query/expansion.py`. Best-effort: falls back to
   `[]` silently on LLM failure, since expansion is a recall booster, not a
   hard requirement for the turn to succeed.
4. **retrieve** — runs dense search (Chroma) and BM25 search over the
   rewritten query *and* every expanded variant. Each retriever's own
   results across all variants are RRF-merged into one ranked list per
   retriever first.
5. **fuse** — `reciprocal_rank_fusion` merges the dense list and the BM25
   list by *rank position*, not raw score — cosine similarity and BM25
   scores live on unrelated scales, so merging by score directly would be
   meaningless.
6. **rerank** — a cross-encoder re-scores the fused candidates down to
   `rerank_top_n`, reading the rank each chunk already held at every prior
   stage to build the `retrieval_proof` shown in the UI.
7. **compress** — trims each surviving chunk to its most query-relevant
   sentences (embedding-based similarity, no LLM call) before it reaches
   the prompt.
8. **generate** — answers using the *original* (not rewritten) question,
   conversation history, and the compressed context. Each source chunk is
   labeled `[Source: <filename>]` in the prompt, and the system prompt
   forbids blending facts across source blocks. Strict grounding: answer
   only from context, or refuse with a fixed verbatim sentence — except a
   narrow carve-out for pure greeting/small-talk messages.
9. **save_turn** — appends this turn to conversation memory for the next
   turn's `rewrite`/`generate` steps to use.
10. **output_guard** *(API layer, not in the graph)* — regex rail re-checks
    the generated answer for PII (catches the model echoing PII that was
    present in a retrieved chunk). Deliberately does not attempt
    jailbreak/unsafe-content detection on the output side — that would
    need semantic judgment, which conflicts with the zero-LLM-call design
    goal (see below).

`GraphState` is a flat `TypedDict(total=False)` with "last write wins"
merge semantics between nodes — see `app/graph/state.py`.

## Key design decisions

- **RRF over raw-score fusion.** Cosine similarity (dense) and BM25 scores
  live on incomparable scales; merging by rank position instead of score
  (`app/retrieval/hybrid_fusion.py`) avoids one retriever silently
  dominating the fused list just because its scores happen to run higher.
- **Deterministic guardrails over LLM self-check.** NeMo Guardrails' common
  pattern asks an LLM to judge every message — a live model call (cost,
  latency, a network dependency) on the critical path of every request.
  This project uses NeMo's built-in `regex` rail instead: same framework,
  zero extra LLM calls, at the honest cost of only catching patterns
  someone thought to add.
- **Thin UI, fat API.** `streamlit_app/` is a pure HTTP client via
  `requests` — it never imports `app/` and ships as its own Docker image.
  The API is the actual product being demonstrated; the UI is a
  presentation layer on top of it, replaceable without touching the core.
- **Explicit prompt-grounding, not implicit.** A soft "answer using only
  the context" instruction let the model hedge and then answer off-topic
  questions from its own training data anyway. Fixed with an exhaustive,
  explicit refusal instruction and a fixed verbatim fallback sentence —
  verified against real off-topic test cases, not assumed fixed
  (`app/graph/nodes.py`'s `generate_node`).
- **Config over conditionals.** Every tunable (chunk size, retrieval `k`,
  the RRF constant, rerank `top_n`, compression sentence count, model
  names) lives in `app/config/settings.py` with a comment explaining why
  that default was chosen — not scattered as magic numbers across call
  sites.
- **Hybrid retrieval's value shows up differently than expected.** On this
  project's small sample corpus, BGE-small dense search alone already
  nails literal keyword/ID matches — RRF's visible contribution here is
  reconciling *runner-up* ranks between retrievers, not rescuing an
  outright miss. A larger, noisier corpus would be needed to show the more
  dramatic "dense fails, hybrid saves it" case — an honest finding, not a
  weakness hidden from the design.
- **Layered LLM fallback, not a single provider.** `llm_gateway.py` tries
  the primary Groq model, then a list of Groq fallback models (only for
  `APIStatusError` — rate limits, decommissioned models — since a dead
  connection fails identically everywhere and shouldn't be retried across
  models), and only falls back to a local Ollama model if every Groq
  option fails. Which model actually answered is returned end-to-end
  (`answer_model`/`answer_provider`) rather than assumed from config.

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | FastAPI + Uvicorn | Routes are sync `def`, not `async def` — guardrails call `asyncio.run()` internally, which breaks inside an already-running event loop; FastAPI runs sync routes in a threadpool instead. |
| AI framework | LangChain + LangGraph | LangGraph drives the 8-node request pipeline (`app/graph/`). |
| Loaders | LangChain loaders (`PyPDFLoader`, `TextLoader`, `UnstructuredWordDocumentLoader`) + `pytesseract` + Groq vision fallback | Dict-dispatched by file extension (`app/loaders/file_loader.py`). |
| Chunking | Recursive (`RecursiveCharacterTextSplitter`) + Semantic (embedding cosine-distance breakpoints) | Recursive is the cheap/deterministic default; semantic catches mid-paragraph topic shifts, slower. |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Asymmetric: query embeddings get a BGE-specific instruction prefix, document embeddings don't. |
| Vector DB | ChromaDB (persistent, cosine-space HNSW) | Single source of truth for the corpus — assigns `chunk_id` and an `active` flag at index time. |
| Sparse retrieval | `rank-bm25` | No storage of its own — rebuilt each process from Chroma's own documents so both retrievers rank identical chunks. |
| Fusion | Reciprocal Rank Fusion (`k=60`, Cormack et al. 2009) | Used twice: dense-vs-BM25 fusion, and collapsing query-expansion variants per retriever. |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB, CPU-friendly, reads query+passage together — more accurate than the bi-encoder, too slow to run over the whole corpus. |
| Query enhancement | Rewrite + expansion (LLM-backed) | `app/query/rewrite.py`, `app/query/expansion.py`. |
| Compression | Embedding-based sentence trimming, no LLM call | `app/compression/context_compressor.py`. |
| LLM gateway | Groq (`llama-3.3-70b-versatile` primary + fallback models) → Ollama (`llama3.2`) | `app/llm/llm_gateway.py`; layered fallback described above. |
| Vision | Groq `qwen/qwen3.6-27b`, Groq-only (no Ollama vision fallback) | Used only when OCR extracts no text from an uploaded image. |
| Guardrails | NeMo Guardrails, built-in `regex` rail only | No LLM self-check call on the critical path — deliberate tradeoff (see Key design decisions). |
| Memory | In-process conversation memory (session + conversation) | Long-term persistence is stubbed (`TODO`) — does not survive a restart. |
| Config | Pydantic Settings + `python-dotenv` | `app/config/settings.py`, grouped by subsystem, every default commented with why. |
| Logging | stdlib `logging` | `app/logging_config/logger.py`. |
| Evaluation | Deterministic proxy metrics, not LLM-as-judge | `app/evaluation/eval_metrics.py` — free, no per-eval model call. |
| Tests | Pytest | Real smoke tests + a per-module `if __name__ == "__main__":` self-test convention throughout `app/`. |
| CI | GitHub Actions (tests only, no deploy step) | `.github/workflows/ci.yml`. |
| Deploy | Docker + Docker Compose (verified) · Render Blueprint (prepared, unverified) | `docker-compose.yml`, `render.yaml`. |
| UI | Streamlit, separate thin client | `streamlit_app/`, its own Dockerfile and `requirements.txt`. |

## API reference

All routes are under `/api/v1`. Every response — including unhandled server
errors — is guaranteed valid JSON via a global exception handler
(`app/main.py`); a startup lifespan hook seeds the 3-document sample corpus
idempotently.

### `POST /api/v1/upload`
Loads, chunks, and indexes a document.
- **Request:** multipart file (`.pdf`/`.txt`/`.docx`/`.png`/`.jpg`/`.jpeg`/`.tiff`/`.bmp`); query param `chunking_strategy` (`recursive` | `semantic`, default `recursive`).
- **Response:** `{filename, chunking_strategy, chunks_indexed}`.
- **Behavior:** dispatches to the right loader by extension. Images run
  OCR first, falling back to a Groq vision description if OCR extracts no
  text (a blank scan, or a genuinely non-text image like an X-ray).
  Invalidates the cached BM25 index so newly uploaded chunks are
  searchable immediately.

### `DELETE /api/v1/documents`
Removes a document from the active corpus.
- **Request:** query params `source` (required), `permanent` (bool, default `true`).
- **Response:** `{source, chunks_deleted, permanent}` — **404** if no chunks match `source`.
- **Behavior:** `permanent=true` hard-deletes the embeddings
  (`VectorStore.delete_by_source`); `permanent=false` soft-deletes by
  flipping an `active` metadata flag (`set_active_by_source`), so the
  document can be restored instantly without re-upload/re-embedding.

### `POST /api/v1/documents/restore`
Un-hides a soft-deleted document.
- **Request:** query param `source`.
- **Response:** `{source, chunks_restored}` — 404 if the source never existed.
- **Behavior:** sets `active=True` again for all of that source's chunks. Restoring an already-active document is a harmless no-op, not a 404.

### `POST /api/v1/chat`
One turn of a (possibly multi-turn) conversation — the full pipeline in [How it works](#how-it-works).
- **Request:** `{query, conversation_id?}` — omit `conversation_id` to start a new conversation; the response echoes back the id to reuse for the next turn.
- **Response:** `{conversation_id, answer, sources, retrieval_count, reranked_count, latency_ms, retrieval_proof, expanded_queries, answer_model, answer_provider}`.
- **Behavior:** `check_input` runs before the graph (400 if blocked), `check_output` runs after (400 if blocked). `retrieval_proof` carries each chunk's BM25/dense/fused/reranked rank for the UI's proof table.

### `POST /api/v1/search`
Retrieval only — no LLM call.
- **Request:** `{query, k}`.
- **Response:** `{results: [{source, text, score}]}`.
- **Behavior:** dense + BM25 → RRF → cross-encoder rerank, the same retrieval stack `/chat` uses, exposed standalone for inspecting retrieval quality in isolation from generation.

### `GET /api/v1/health`
Liveness check.
- **Response:** `{status, app_env}` — deliberately touches no dependencies (Chroma, embeddings, LLM) so it stays fast and meaningful as a pure "is the process up" signal.

### `GET /api/v1/stats`
Corpus introspection for the UI's sidebar.
- **Response:** `{document_count, chunk_count, sources, hidden_sources}`.

## Running it

### Docker Compose (recommended)

```bash
docker compose up --build
# API:       http://localhost:8000/api/v1/health
# Streamlit: http://localhost:8501
```

`docker-compose.yml` runs two services: `app` (API, port 8000, volumes for
`./data` and the host's Hugging Face cache) and `streamlit` (UI, port 8501,
`API_BASE_URL=http://app:8000` via Compose's built-in service-name DNS,
`depends_on: app`). No database or queue beyond these two containers.

### Local dev (one venv, two processes)

```bash
python -m venv venv && source venv/Scripts/activate   # venv/bin/activate on Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload                          # API on :8000

# in a second terminal:
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py                                   # UI on :8501
```

### Environment

Copy `.env.example` to `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=
CHROMA_PERSIST_DIR=./data/chroma_db
TESSERACT_CMD=/usr/bin/tesseract
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
APP_ENV=development
LOG_LEVEL=INFO
```

`GROQ_API_KEY` is free-tier (console.groq.com). Without it, set
`LLM_PROVIDER=ollama` and run a local Ollama server instead — the gateway
also falls back to Ollama automatically at runtime if every configured
Groq model fails. `TESSERACT_CMD` defaults to the Linux path used by
Docker; on Windows, install Tesseract locally and point this at the real
binary path instead.

## Configuration

All tunables live in `app/config/settings.py` (Pydantic Settings, loaded
from `.env`), grouped by subsystem. Every default has an inline comment
explaining the choice — this table is a map to that file, not a
replacement for reading it.

| Group | Key settings | Purpose |
|---|---|---|
| LLM gateway | `llm_provider`, `groq_api_key` | Which provider answers by default; Groq is free-tier/fast, Ollama is the local fallback. |
| Vector store | `chroma_persist_dir`, `chroma_collection_name` | Where/how the persistent Chroma index is stored. |
| OCR | `tesseract_cmd` | Path to the Tesseract binary (differs between Docker/Linux and local Windows dev). |
| Vision | `groq_vision_model`, `groq_vision_fallback_models` | Model used only when OCR extracts no text from an image upload. |
| Embeddings | `embedding_model`, `bge_query_instruction` | Embedding model and BGE's asymmetric query-side instruction prefix. |
| Chunking | `chunk_size`, `chunk_overlap`, `semantic_breakpoint_percentile` | Recursive splitter sizing and the semantic chunker's topic-shift threshold. |
| Retrieval | `retrieval_top_k`, `rrf_k` | How many candidates each retriever surfaces pre-fusion, and RRF's smoothing constant. |
| Query LLM tools | `groq_model`, `groq_fallback_models`, `ollama_model`, `query_expansion_n` | Model(s) used for rewrite/expand/generate, and how many expansion variants to generate. |
| Rerank | `cross_encoder_model`, `rerank_top_n` | Which cross-encoder, and how many chunks survive reranking into the final context. |
| Compression | `compression_sentences_per_doc` | How many sentences per reranked chunk survive into the prompt. |
| Memory | `conversation_memory_max_turns` | How many past turn-pairs feed back into rewrite/generate each turn. |
| API | `upload_dir` | Where uploaded files land before loading/chunking. |
| App | `app_env`, `log_level` | Environment tag and log verbosity. |

## Testing

```bash
pytest -q
```

Real smoke tests, not exhaustive behavioral coverage, in `tests/`:
- `conftest.py` points Chroma at a throwaway temp directory/collection so tests never touch the real persistent index.
- `test_api.py` — health, stats, an upload→search round trip, rejecting a bad file extension, a grounded chat answer, and blocking a jailbreak attempt.
- `test_loaders.py` — text loading (happy path, missing file, bad extension), and OCR image loading.
- `test_retrieval.py` — vector store add+search, BM25 exact-keyword matching, RRF merge, and cross-encoder rerank reordering.

Two tests are environment-conditional and skip cleanly rather than fail when
their dependency isn't available locally: the OCR test needs a real
`tesseract` binary, and the live chat test needs `GROQ_API_KEY` set. Both
run for real in CI (`.github/workflows/ci.yml`, which installs
`tesseract-ocr` and reads `GROQ_API_KEY` from a repo secret) and in Docker.

Beyond `tests/`, most modules under `app/` also carry their own runnable
`if __name__ == "__main__":` self-test — a consistent pattern throughout
the codebase, useful for exercising one module in isolation (e.g.
`python -m app.retrieval.hybrid_fusion`) without booting the full API.

## Deployment

### Docker Compose — verified working

`docker compose up --build` runs both services locally end-to-end; this is
the primary, tested deployment path (see [Running it](#running-it)).

### Render — prepared, not yet deployed

`render.yaml` defines a Blueprint with two Docker-runtime web services
(`unirag-api`, `unirag-ui`); connecting the repo in Render's dashboard would
auto-deploy both on every push to `main`, with no GitHub Actions involvement
in the deploy step itself — CI (`.github/workflows/ci.yml`) is a quality
gate only. Honest status: **this has not been deployed or verified against
a live Render account.** Known/flagged issues going in:
- `GROQ_API_KEY` is `sync: false` — entered manually in Render's dashboard, never committed, by design.
- Render's free tier has an ephemeral filesystem; a commented-out `disk:` block exists for `unirag-api` to persist `CHROMA_PERSIST_DIR` across restarts — uncomment it if the demo needs uploaded documents to survive a redeploy.
- `unirag-ui`'s `API_BASE_URL` uses `fromService`/`property: host` to auto-wire to `unirag-api`'s address — this is **unverified** against a live Render account; the file's own comments flag it as something to check after first deploy (it may resolve to a bare hostname rather than a working `https://` URL, in which case it needs to be set manually).

## Project structure

```
UniRAG/
├── app/
│   ├── main.py            FastAPI app: routes, lifespan seed, global exception handler
│   ├── config/             settings.py — every tunable, one place, commented
│   ├── loaders/             file_loader.py (pdf/txt/docx); ocr_loader.py (OCR + vision fallback)
│   ├── chunking/            recursive_chunker.py, semantic_chunker.py
│   ├── embeddings/          bge_embedder.py — lazy singleton, asymmetric query/doc encoding
│   ├── retrieval/           vector_store.py (Chroma), bm25_retriever.py, hybrid_fusion.py (RRF)
│   ├── query/               rewrite.py, expansion.py
│   ├── rerank/              cross_encoder_rerank.py
│   ├── filtering/           metadata_filter.py — built, not yet wired into the pipeline
│   ├── compression/         context_compressor.py
│   ├── graph/               state.py, nodes.py, build_graph.py — the LangGraph pipeline
│   ├── guardrails/          input_guard.py, output_guard.py — NeMo regex rails
│   ├── memory/              conversation_memory.py, session_memory.py
│   ├── llm/                 llm_gateway.py — Groq/Ollama text + Groq vision gateway
│   ├── evaluation/          eval_metrics.py — deterministic proxy metrics
│   └── logging_config/      logger.py
├── streamlit_app/           thin HTTP-only UI; own Dockerfile + requirements.txt
├── tests/                   pytest suite + conftest.py (isolated Chroma temp dir)
├── data/                    chroma_db/ (persisted index), uploads/ (raw files) — gitignored
├── .github/workflows/ci.yml tests-only CI, no deploy step
├── Dockerfile                API image
├── docker-compose.yml         app + streamlit services
├── render.yaml                Render Blueprint (API + UI) — prepared, unverified
├── requirements.txt            API dependencies (unpinned)
├── .env.example                settings template
└── docs/CLAUDE.md                   full build spec + day-by-day progress log (§12)
```

## Build process & interview talking points

Built solo over a 6-day plan (one module group per day) under two hard
constraints: free/open-source tools only (Groq's free tier, no paid APIs),
and legibility over completeness. Deliberately dropped to stay in scope:
LiteLLM, FAISS, Guardrails AI, OpenAI embeddings, MMR retrieval — full
reasoning for each in `docs/CLAUDE.md` §2-3.

A few of the more interesting things found while building it:
- **Hybrid retrieval's benefit was subtler than expected in practice** — on
  this project's small sample corpus, dense search alone already handles
  literal keyword/ID matches; RRF's visible value here is reconciling
  runner-up ranks, not rescuing an outright miss (see Key design
  decisions).
- **A "grounded" model still answered off-topic questions from training
  data** until an explicit, exhaustive refusal instruction replaced a soft
  one — verified against real test cases (`app/graph/nodes.py`).
- **A week of intermittent "network/cert block" symptoms turned out to be
  Avast's HTTPS-scanning feature** intercepting TLS to `api.groq.com`/
  `huggingface.co` with its own certificate — trusted by Windows tools, not
  by Python's `certifi` store. Nothing was wrong with the code, Groq, or
  the network. Full story in `docs/CLAUDE.md`'s Day 8 log and environment notes.
- **Two real test-writing bugs, not just app bugs** — a BM25 IDF=0 edge
  case on a 2-document corpus needed a 3rd distractor document to expose
  correctly; the OCR test needed a `skipif` gate since `tesseract` isn't
  installed on every dev machine.

For a deeper walkthrough of any one mechanism, these modules each carry an
inline "why this works" comment and interviewer-style questions, per this
project's own documentation convention:

| Topic | File |
|---|---|
| RRF fusion | `app/retrieval/hybrid_fusion.py` |
| Query rewrite/expansion prompts | `app/query/rewrite.py`, `app/query/expansion.py` |
| Cross-encoder rerank | `app/rerank/cross_encoder_rerank.py` |
| LangGraph structure | `app/graph/build_graph.py` |
| Guardrail tradeoffs | `app/guardrails/input_guard.py`, `app/guardrails/output_guard.py` |

Full day-by-day build log, every deviation from the original plan (with
reasoning), and environment notes live in `docs/CLAUDE.md` §12.

## Known limitations

- **No streaming** — `/chat` returns the full answer in one blocking response.
- **Long-term conversation memory is in-process only** (stubbed with a `TODO` in `app/memory/conversation_memory.py`) — doesn't survive a restart.
- **No per-domain/per-tenant data isolation** — every upload goes into one shared corpus; there's no `domain` tag on documents and retrieval doesn't filter by one. Two options are on record, neither built yet: a fully separate UniRAG deployment per downstream consumer (simplest, zero code), or wiring the already-built-but-unused `app/filtering/metadata_filter.py` into upload/retrieval so one shared instance can tag and filter by domain. Deferred until a second real consumer actually needs it, not forgotten.
- **Render deployment is unverified** — `render.yaml` is prepared but has not been deployed against a live account; `unirag-ui`'s `API_BASE_URL` wiring in particular is flagged as needing a manual check after first deploy.
- **BM25 is rebuilt from the full corpus on each process start / after every upload**, not incrementally maintained — fine at this project's scale, would need revisiting for a larger corpus or multi-process deployment.
- **Local Chroma's persistent store isn't safe under concurrent multi-process access** — a real corpus-consistency incident during development was traced to multiple live processes holding simultaneous connections to the same persist directory; if corpus data looks inconsistent, check for stray duplicate processes before assuming a code bug.

## License

MIT — see [`LICENSE`](./LICENSE).

## Further reading

- `docs/CLAUDE.md` — the original build spec, hard constraints, and the full day-by-day progress log (§12) this README's "Build process" section summarizes.
- `PROJECT_SUMMARY.md` — a fuller narrative version of the same build history, written for a portfolio audience.
- `UI_UPGRADE_SPEC.md` — the UI's visual design rationale (dark "technical/lab" aesthetic, palette, why custom CSS was chosen over a framework migration).
