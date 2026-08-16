# CLAUDE.md — UniRAG

This file gives Claude Code the standing context for this repo. Read it before
writing any code. If something here conflicts with a request in chat, the chat
request wins — but say out loud that you're deviating.

---

## 1. What this project is

UniRAG is a **reusable AI Knowledge Core** — a production-shaped Retrieval-Augmented
Generation platform that will later be the retrieval backbone for two downstream
copilots (Engineering AI Copilot, Medical AI Copilot).

```
                AI Knowledge Core (UniRAG)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  Engineering AI Copilot       Medical AI Copilot
```

**Consequence for you (Claude Code):** every module must be importable and usable
standalone. No hard-coded domain assumptions, no "engineering" or "medical" strings
anywhere in `app/`. Domain-specific behaviour comes from config and metadata only.

This is a **portfolio project built to be explained in interviews**, not a product.
Optimise for *legibility and defensibility*, not for cleverness or completeness.

---

## 2. Hard constraints

- **Everything must be free / open-source.** No paid APIs beyond Groq's free tier.
- **6-day build, one day per module group.** Do not build ahead of the current day.
- **Do not add dependencies** outside `requirements.txt` without asking first.
- **Do not install or use:** LiteLLM, FAISS, Guardrails AI, OpenAI/MiniLM embeddings,
  MMR retrieval. These were deliberately dropped.
- Python 3.11+. Local dev first; Docker on Day 6.

---

## 3. Tech stack (fixed — do not substitute)

| Layer | Choice |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI framework | LangChain + LangGraph |
| Loaders | LangChain loaders; Tesseract via `pytesseract` for images |
| Chunking | Recursive + Semantic |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| Vector DB | ChromaDB (persistent, `./data/chroma_db`) |
| Retrieval | BM25 + dense, fused with **RRF**; metadata filtering; context compression; cross-encoder rerank |
| Query enhancement | Rewrite + expansion |
| LLM gateway | Manual switch in `app/llm/llm_gateway.py` — Groq default, Ollama fallback |
| Guardrails | NeMo Guardrails only |
| Memory | Session + conversation (long-term = stub with a clear `TODO`) |
| Config | Pydantic Settings + `python-dotenv` |
| Logging | stdlib `logging` |
| Tests | Pytest, **smoke tests only** |
| Deploy | Docker + Docker Compose, Render (Day 7) |
| UI | Streamlit (Day 7, separate, thin) |

---

## 4. Folder structure (create files only in these paths)

```
unirag/
├── app/
│   ├── main.py                     # FastAPI entrypoint (Day 6)
│   ├── loaders/       file_loader.py, ocr_loader.py                 # Day 1
│   ├── chunking/      recursive_chunker.py, semantic_chunker.py     # Day 1
│   ├── embeddings/    bge_embedder.py                               # Day 2
│   ├── retrieval/     vector_store.py, bm25_retriever.py,
│   │                  hybrid_fusion.py                              # Day 2
│   ├── query/         rewrite.py, expansion.py                      # Day 3
│   ├── rerank/        cross_encoder_rerank.py                       # Day 3
│   ├── filtering/     metadata_filter.py                            # Day 3
│   ├── compression/   context_compressor.py                         # Day 4
│   ├── graph/         state.py, nodes.py, build_graph.py            # Day 4
│   ├── memory/        session_memory.py, conversation_memory.py     # Day 4
│   ├── guardrails/    input_guard.py, output_guard.py               # Day 5
│   ├── evaluation/    eval_metrics.py                               # Day 5
│   ├── logging_config/ logger.py                                    # Day 5
│   ├── llm/           llm_gateway.py                                 # already written
│   └── config/        settings.py
├── tests/             test_loaders.py, test_retrieval.py, test_api.py
├── data/              uploads/, chroma_db/          # gitignored
├── streamlit_app/     app.py                        # Day 7
└── .env, .env.example, requirements.txt, Dockerfile,
    docker-compose.yml, .dockerignore, .gitignore, README.md
```

**Do not create new top-level folders or "utils" dumping grounds.** If something
doesn't fit, ask.

---

## 5. Code style rules (important — these are non-negotiable)

1. **One concern per file.** Each module in the tree above should be readable
   top-to-bottom in one sitting.
2. **Heavy inline comments explaining the *why*, not the *what*.**
   - Bad: `# loop through chunks`
   - Good: `# RRF uses rank, not score, so BM25 and dense scores never need
     normalising to a common scale — this is the whole point of the algorithm`
3. Every public function gets a short docstring: what it takes, what it returns,
   and one line on when you'd use it.
4. **Type hints everywhere.** Pydantic models for anything crossing a boundary.
5. No magic numbers. Chunk sizes, `k` values, RRF constant, rerank top-n — all in
   `app/config/settings.py` with a comment on why that default was chosen.
6. Every module ends with a `if __name__ == "__main__":` block that runs a tiny
   self-test on sample input. This is how the day's finish line gets verified.
7. No secrets in code. Read from `.env` via settings.
8. Fail loudly at the edges (bad file type, empty retrieval), quietly in the middle.
9. Keep functions under ~40 lines. If it's longer, it's doing two things.

---

## 6. Division of labour — you vs me

Some parts I need to *own* for interviews. For those, don't just write it and move on.

| Component | Who drives | Your job |
|---|---|---|
| Loaders, OCR, chunking | You | Write it, explain chunking tradeoffs briefly |
| Chroma setup, BM25 wiring | You | Write it |
| **RRF fusion (`hybrid_fusion.py`)** | **Me** | Write a clear reference version, then explain the formula line by line and give me 2 questions an interviewer would ask |
| Query rewrite/expansion prompts | Me | Draft, then explain why each prompt is shaped that way |
| Cross-encoder rerank logic | Me | Same |
| **LangGraph structure (`graph/`)** | **Me** | Write nodes/edges, then produce an ASCII diagram of the graph I can redraw on a whiteboard |
| Guardrail rule logic | Me | Draft rules, explain what each catches and what it misses |
| FastAPI routes, Dockerfile, compose | You | Write it, one-line comment per service |
| Logging + eval scaffolding | You | Write it |

For anything marked **Me**: after the code, add a short "Why this works" note
(5–10 lines) in the file's module docstring.

---

## 7. Day-wise plan and finish lines

Work one day at a time. **Stop at the finish line and report.** Don't start the
next day's files unprompted.

| Day | Build | Finish line |
|---|---|---|
| 1 | Loaders + OCR + chunking | Upload one PDF and one image → clean chunks printed |
| 2 | Embeddings + Chroma + BM25 + RRF | A query returns relevant chunks; hybrid beats pure dense on a keyword-heavy query |
| 3 | Query rewrite/expansion, metadata filter, rerank | Rewritten query improves retrieval; filter narrows; rerank visibly reorders top-k |
| 4 | Compression + LangGraph + memory | One multi-turn conversation works end-to-end, memory carries context |
| 5 | Guardrails + eval + logging | Malicious/PII prompts caught; every request logs latency, retrieval count, sources |
| 6 | FastAPI + Docker | `docker-compose up` → all 4 endpoints respond → one upload→chat round trip in a container |
| 7 | Streamlit + Render deploy (optional) | Live demo URL |

**Day 4 is the highest-risk day.** If we're running behind, protect Day 4 and cut
scope elsewhere.

API surface (Day 6): `POST /api/v1/upload`, `POST /api/v1/chat`,
`POST /api/v1/search`, `GET /api/v1/health`.

---

## 8. Working protocol

- **Before writing code for a new day:** state in 3–5 lines what you're about to
  build and how the pieces connect. Wait for my go-ahead if it differs from this file.
- **Change one thing at a time.** No sweeping refactors across modules unless I ask.
- **Don't silently fix unrelated code** you notice in passing — mention it instead.
- **After each file:** run its `__main__` self-test and show me the actual output.
  Don't claim it works without running it.
- **When you hit an error:** show me the real traceback before proposing a fix.
  Don't paper over it with a try/except.
- **When something in this plan is genuinely a bad idea, say so.** I'd rather
  redesign on Day 2 than discover it on Day 6.
- **At the end of each day:** write a short summary — what got built, how the pieces
  fit together, one analogy for the core idea of the day, and what's still stubbed.

---

## 9. Commands

```bash
# setup
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# module self-test
python -m app.retrieval.hybrid_fusion

# tests
pytest -q

# run API (Day 6+)
uvicorn app.main:app --reload

# containers (Day 6+)
docker-compose up --build
```

---

## 10. Environment

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=
CHROMA_PERSIST_DIR=./data/chroma_db
TESSERACT_CMD=/usr/bin/tesseract
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
APP_ENV=development
LOG_LEVEL=INFO
```

Keep `.env.example` in sync whenever a new setting is added. `.env`, `data/uploads/`,
and `data/chroma_db/` stay gitignored.

---

## 11. Definition of done (v1)

- [ ] All 6 days' finish lines pass
- [ ] `docker-compose up` gives a working upload→chat round trip
- [ ] README explains architecture with a diagram and one "why this design" section
- [ ] Every "Me"-owned module has a "Why this works" note in its docstring
- [ ] No domain-specific logic anywhere in `app/`

---

## 12. Progress log

### Day 2 — Embeddings + Chroma + BM25 + RRF (done)

Built `app/embeddings/bge_embedder.py`, `app/retrieval/vector_store.py`,
`app/retrieval/bm25_retriever.py`, `app/retrieval/hybrid_fusion.py`. New settings:
`chroma_collection_name`, `bge_query_instruction`, `retrieval_top_k`, `rrf_k`.

How the pieces fit: `VectorStore` is the one source of truth for the corpus
(Chroma, persisted, cosine space) — it assigns each chunk a `chunk_id` on
add. `BM25Retriever` doesn't keep its own copy of the documents; it's built
from `VectorStore.get_all_documents()` so both retrievers always rank the
same chunks. `reciprocal_rank_fusion()` merges their two ranked lists by
*position*, not raw score, keyed on that shared `chunk_id`.

Analogy: two judges scoring a contest with completely different scoring
systems (one out of 10, one out of 100) — instead of arguing whose scale is
"real," you just ask each judge for their ranking and combine the rankings.

What the finish line actually showed: on a small clean corpus, BGE-small
turned out to be good enough at literal keyword/ID matching that dense
retrieval alone already found the right document (see `hybrid_fusion.py`'s
self-test and its docstring note) — RRF's visible value was in reordering
the *runner-up* candidates using both retrievers' evidence, not rescuing a
missed top result. Worth remembering this if a demo later needs "dense
fails, hybrid saves it" — that scenario needs a bigger/noisier corpus than
a handful of documents to actually happen.

### Day 3 — Query rewrite/expansion, metadata filter, rerank (done)

Built `app/query/rewrite.py`, `app/query/expansion.py`,
`app/filtering/metadata_filter.py`, `app/rerank/cross_encoder_rerank.py`. New
settings: `groq_model`, `ollama_model`, `query_expansion_n`,
`cross_encoder_model`, `rerank_top_n`.

**Deviation from §7 (flagged and approved in chat):** `app/llm/llm_gateway.py`
was empty despite this file annotating it "already written," and Day 3's
rewrite/expansion genuinely need a real LLM call to do anything. Wrote a
minimal `get_completion()` now (Groq default / Ollama fallback) rather than
stub it out or go heuristic-only, ahead of its nominal Day 6 slot.

How the pieces fit: `rewrite.py` and `expansion.py` both call
`llm_gateway.get_completion()` with different system prompts (one query in
→ one clean query out; one query in → N phrasings out).
`metadata_filter.py` has two entry points because Chroma can filter
natively (`where` clause, pushed down before the query runs) but BM25/RRF
results can't, so they get filtered after the fact on the same metadata.
`cross_encoder_rerank.py` runs last, re-scoring the small fused candidate
set with a slower but more accurate model that reads the query and passage
together, instead of comparing two independently-computed vectors.

Analogy: `rewrite`/`expansion` are like a reference librarian cleaning up
and multiplying your question before you go looking; `cross_encoder_rerank`
is like a final human reviewer reading the short stack of books someone
already pulled for you and putting the actually-relevant one on top.

Bug found and fixed during verification: the Ollama fallback in
`llm_gateway.py` was passing `settings.groq_model` (a Groq-only model ID)
instead of a proper Ollama tag — added `ollama_model` as its own setting.

### Day 4 — Compression + LangGraph + memory (done)

Built `app/compression/context_compressor.py`, `app/memory/conversation_memory.py`,
`app/memory/session_memory.py`, `app/graph/state.py`, `app/graph/nodes.py`,
`app/graph/build_graph.py`. New settings: `compression_sentences_per_doc`,
`conversation_memory_max_turns`.

How the pieces fit: `GraphState` is a flat `TypedDict` — each node reads the
keys it needs and returns only the keys it changed, LangGraph merges the
rest forward untouched. The graph itself is linear:
`rewrite → retrieve → fuse → rerank → compress → generate → save_turn`.
`rewrite_node` is where memory actually enters the pipeline: it pulls prior
turns from `conversation_memory.get_history()` and, if any exist, prepends
them to the raw query before calling `rewrite.rewrite_query()` unchanged —
that function's existing prompt already resolves vague references "if the
query provides enough context," so feeding it conversation history was
enough to make pronoun resolution work without touching Day 3 code.
`generate_node` answers using the *original* question (not the rewritten
one) plus history plus the compressed context, then `save_turn_node` records
the exchange so the next turn has it as history. `context_compressor.py`
reuses `BGEEmbedder` (no LLM call) to keep only each reranked chunk's most
query-relevant sentences, in their original order.

Analogy: the graph is an assembly line, and `GraphState` is the tray that
rides along it — each station only looks at and changes the parts of the
tray it cares about, never the whole thing.

Finish line, verified for real: ran two turns of one conversation through
the compiled graph. Turn 2 asked "Why does *it* use rank instead of raw
score?" — rewrite resolved "it" into "...in Reciprocal Rank Fusion" using
turn 1's history, and the final answer stayed correct and coherent with
turn 1's topic. Full transcript is in the session, not reproduced here.

**Scope cut, flagged per CLAUDE.md's own rule:** `app/query/expansion.py`'s
multi-query capability is written (Day 3) but not wired into this graph.
`reciprocal_rank_fusion` already takes an arbitrary number of ranked lists,
so adding expansion later means retrieving on each variant and passing all
lists into fusion — additive, not a redesign. Left out now to protect Day 4
per CLAUDE.md §7's own "if running behind, cut scope elsewhere" guidance.

**Known limitation, not fixed today:** `nodes.py`'s BM25 index is built once
per process (lazy singleton) from whatever's in Chroma at that moment — it
does not refresh if documents are added mid-process. Fine for a single-run
demo; a real deployment would rebuild it on ingest. Worth revisiting around
Day 6 if the API needs to serve uploads and chat concurrently.

### Day 5 — Guardrails + eval + logging (done)

Built `app/guardrails/input_guard.py`, `app/guardrails/output_guard.py`,
`app/logging_config/logger.py`, `app/evaluation/eval_metrics.py`. Also added
one additive helper, `run_turn()`, to `app/graph/build_graph.py` (Day 4 file
— flagged and approved) so logging could be proven against a real request
instead of only a synthetic self-test.

**Design decision, discussed and approved in chat before writing code:**
NeMo Guardrails' usual "self-check" rails ask an LLM to judge every message,
which would put a live LLM call on the critical path of every request and
inherit the same Groq network fragility hit on Day 3/4. Chose instead to use
NeMo Guardrails' *built-in* `regex` library rail (`regex check input` /
`regex check output` — ships with the package, zero custom Colang needed) —
still the real `RailsConfig`/`LLMRails` engine, but the actual detection is
deterministic pattern matching, not an LLM judgment call. `RailsConfig` still
requires a `models:` entry to instantiate, so it's pointed at Ollama — but
`check_async(..., rail_types=[RailType.INPUT/OUTPUT])` runs only the named
rail and never actually calls that model for these checks.

How the pieces fit: `input_guard.PII_PATTERNS` (email/SSN/credit-card/phone
shaped regexes) and `input_guard.JAILBREAK_PATTERNS` (known jailbreak
phrasings) both feed NeMo's regex rail via an inline YAML config
(`RailsConfig.from_content` — no new config files/folders). `output_guard.py`
reuses `PII_PATTERNS` (not a second copy) against the *generated answer*,
on the theory that a RAG system's real output-leakage risk is the model
echoing PII that was sitting in a retrieved chunk, not inventing unsafe
content from nothing. `logger.py` is one stdlib logger + a `log_request()`
helper with a fixed message shape; `run_turn()` in `build_graph.py` is the
single seam every request passes through, so it's the one place that times
and logs rather than instrumenting every node individually. `eval_metrics.py`
keeps to two free, deterministic proxies — `retrieval_hit_rate` (against a
hand-labeled eval set) and `context_overlap_ratio` (a word-overlap
groundedness heuristic) — explicitly not an LLM-as-judge, to stay within
"smoke tests only."

Analogy: the guardrails are a metal detector, not a mind-reader — fast,
free, and certain about what it's built to detect (a shape it recognizes),
and just as certain to miss anything shaped differently.

Finish line, verified for real: `input_guard` blocked a jailbreak phrase and
a PII-bearing query while passing a clean one; `output_guard` blocked an
answer that echoed an email address; `run_turn()`'s log output showed real
per-request `latency_ms`, `retrieval_count`, and `sources` for both turns of
Day 4's demo conversation.

**What each guardrail catches and misses (the "Me"-owned explanation, kept
here too since it's the kind of thing worth having on hand for an
interview):** see the module docstrings in `input_guard.py`/`output_guard.py`
directly — they include the full catches/misses breakdown and two
interviewer-style questions each, per CLAUDE.md §6.

### Day 6 — FastAPI + Docker (done)

Built `app/main.py` (the four API-surface endpoints from CLAUDE.md §7),
`Dockerfile`, `docker-compose.yml`, `.dockerignore`. New setting: `upload_dir`.

**Design decision, discussed and approved in chat before writing code:** this
dev machine's network blocks HTTPS to huggingface.co and api.groq.com (see
below) — inside Docker, that would break both model loading and the primary
LLM path. Chose to mount the host's HF cache as a volume (`HF_HUB_OFFLINE=1`)
and default `LLM_PROVIDER=ollama` (reaching the host's Ollama via
`host.docker.internal`) in `docker-compose.yml` specifically, so the finish
line could be verified live on this machine — `.env` itself still specifies
Groq as CLAUDE.md's real default; the compose overrides are dev-machine-only.

How the pieces fit: every route in `main.py` is a plain `def`, not
`async def` — not a style choice, a correctness one. `input_guard`/
`output_guard` call `asyncio.run()` internally (see their Day 5 docstrings),
which raises if called from inside an already-running event loop; FastAPI
runs sync `def` routes in a worker thread instead, where that's safe. The
graph, `VectorStore`, and `BM25Retriever` are built once per process (lazy
singletons, same pattern as `app/graph/nodes.py`), not once per request.
Guardrails are wired in at the route level in `/chat` (`check_input` before
`run_turn`, `check_output` after) — `build_graph.py`'s nodes/edges are
untouched, exactly as planned after Day 5. `/upload` writes to a UUID-named
temp path (avoids filename collisions) but resets `metadata["source"]` back
to the original filename after loading, and invalidates the BM25 singleton
so the next `/search` or `/chat` rebuilds it from the newly larger corpus.
`/search` is retrieval-only (dense + BM25 + RRF + rerank, no LLM call) — a
separate, cheaper path than `/chat` for inspecting what retrieval finds.

Analogy: `main.py` doesn't do any RAG work itself — it's a receptionist that
routes each request to the right already-built department (upload → loaders/
chunking/vector_store; chat → guardrails + the Day 4 graph; search → just
retrieval) and translates whatever comes back into an HTTP response.

**A second, smaller network wall hit and fixed during this day:** even with
the Ollama/offline-HF workaround, `docker build` itself failed —
`pip install` from inside the container couldn't reach pypi.org
(`SSLCertVerificationError: unable to get local issuer certificate`), even
though the *host's* pip installs were unaffected all session. Same class of
problem as the huggingface.co/api.groq.com block, just hitting a third host
only once traffic went through Docker's network path. Fixed with
`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org`
in the Dockerfile — scoped to package installation at build time, doesn't
touch the app's own runtime HTTPS calls.

Finish line, verified for real inside the running container (not just
locally via TestClient): `GET /api/v1/health` → 200; `POST /api/v1/upload`
→ indexed a chunk; `POST /api/v1/search` → returned reranked results;
`POST /api/v1/chat` → 200 with a grounded answer, and a jailbreak-phrased
request → 400 `{"error": "input_blocked", "reason": "jailbreak_attempt"}`;
container logs showed real `latency_ms`/`retrieval_count`/`sources` for the
chat request, proving Day 5's logging finish line holds inside Docker too.

### Day 7 — Streamlit UI (done; Render deploy not attempted)

Built `streamlit_app/app.py`, `requirements.txt`, `Dockerfile`; added it as a
second service in `docker-compose.yml`. One small backend addition:
`ChatResponse` now also returns `retrieval_count`, `reranked_count`, and
`latency_ms` (real numbers already computed in `run_turn()`, just not
returned to the caller before). Also added `GET /api/v1/stats` (document/
chunk counts) — a 5th endpoint beyond §7's literal 4-endpoint list, added
specifically so the UI could show real corpus size instead of a placeholder.

**Two UI mockups were discussed before building, not just built from
directly** — worth remembering the reasoning, not just the pixels:
- v1 (medical-branded SaaS dashboard) was rejected: conflicts with §1's
  domain-neutrality rule (UniRAG is the core, "Medical Copilot" is a
  separate downstream consumer) and isn't a "thin" Streamlit shape.
- v2 (dark, monospace-accented, pipeline-trace-focused) became the actual
  reference — its "Domain: General/Engineering/Medical" selector encodes
  the copilot architecture as a real feature instead of conflicting with
  it, and its meta-row (latency/chunks/guardrails) maps onto data the API
  already computes. Built as **visual-only** for the Domain selector
  (confirmed in chat) — it doesn't filter retrieval; that's a clean, small
  follow-up later via Day 3's existing `metadata_filter.py`, once there's
  a real reason to tag documents by domain.
- The pipeline trace (`rewrite → retrieve → fuse → rerank → compress →
  generate`) is shown **after** the answer arrives, all stages marked
  done — not live/animated stage-by-stage, since `/api/v1/chat` is one
  blocking call, not a stream. True live progress would need restructuring
  the endpoint (SSE/WebSocket), which is backend work, not thin-UI work.

How the pieces fit: `streamlit_app/app.py` never imports from `app/` — every
action is a `requests` call to the FastAPI service, which is also why it's
a separate image with its own `requirements.txt` (just `streamlit` +
`requests`, none of the ML dependencies). Sidebar nav uses
`st.segmented_control`; stat cards use native `st.metric`; the pipeline
trace and meta-row are the only hand-styled CSS on the page (a handful of
`<span class="stage">` pills), everything else is native Streamlit
components. In `docker-compose.yml`, the `streamlit` service reaches `app`
over Docker's built-in service-name DNS (`API_BASE_URL=http://app:8000`),
not `localhost` or `host.docker.internal` — those are for reaching the
*host* machine (Ollama), this is container-to-container.

Analogy: `streamlit_app/app.py` is a remote control, not a second copy of
the TV — it sends button-presses (HTTP requests) to the one real device
(`app/main.py`) and displays whatever comes back; it has no picture tube of
its own.

**A real, unrelated bug found through manual testing, not code review:**
asking the chat endpoint an off-topic question ("write me a python function
to reverse a string") produced *"Context does not provide information...
However, here is a simple Python function..."* — the model hedged, then
answered anyway from its own training data, defeating the entire point of
"answers grounded in your own documents." Root cause: `generate_node`'s
system prompt (`app/graph/nodes.py`, Day 4) said "answer using only the
provided context... if the context doesn't contain the answer, say so
instead of guessing" — soft enough that a small local model (`llama3.2:3b`
via Ollama) would acknowledge it and then ignore it. Fixed by making the
refusal explicit and exhaustive (spells out *what not to do* — no code, no
problem-solving, no general knowledge — and gives the exact refusal
sentence to output verbatim, no caveat-then-answer allowed). Verified
against three cases post-fix: an off-topic code request, an off-topic
general-knowledge question ("capital of France"), and a genuine in-corpus
question — the first two now refuse cleanly, the third still answers
normally. This is a prompt-grounding gap, not a guardrails gap:
`input_guard`/`output_guard` only check PII/jailbreak patterns (Day 5
scope) and correctly let this through both times — grounding to context is
`generate_node`'s job, not theirs.

Finish line ("live demo URL"), status: the *local* half is fully verified —
`docker-compose up` brings up both services, Streamlit reaches the API over
Docker's internal network, upload → chat → refusal-on-off-topic all work
in the containerized stack. The *deploy* half (Render, a public URL) was
**not attempted** — CLAUDE.md marks all of Day 7 optional, and a Render
deploy needs the user's own account/credentials plus creates real
internet-facing infrastructure (and would expose `GROQ_API_KEY` as a
platform env var) — outside what should happen without the user directly
driving it.

### Day 8 — Post-launch hardening: LLM resilience, document lifecycle, image understanding (ongoing)

Not a single day's build — this is an extended live-testing session after all
7 days were nominally "done," where the user manually drove the running app
and reported real failures as they hit them. Kept as one dated entry per
CLAUDE.md's own logging convention, covering everything found and fixed.

**LLM gateway resilience (`app/llm/llm_gateway.py`).** Real usage exposed
that `LLM_PROVIDER=groq` failing meant a human manually restarting the
process with `LLM_PROVIDER=ollama` every time — happened repeatedly in one
session. Fixed in two layers: (1) `_call_groq` now tries
`groq_fallback_models` (multiple Groq models in order) before giving up on
Groq, catching only `APIStatusError` (Groq responded — rate limit,
decommissioned model) and deliberately not `APIConnectionError` (no
response at all — retrying other models can't fix a dead connection); (2)
`get_completion` now falls back to Ollama automatically if every Groq model
fails, instead of raising. A new `Completion` dataclass (`text`, `model`,
`provider`) and `get_completion_with_model()` make *which* model actually
answered checkable from the API/UI (`ChatResponse.answer_model`/
`answer_provider`, rendered in the Streamlit meta-row) instead of assumed
from `settings.groq_model` — necessary now that the answering model isn't
fixed. Primary model bumped from `llama-3.1-8b-instant` to
`llama-3.3-70b-versatile` on direct request (better quality, slower/costlier
— including for rewrite/expand, which share this same setting; flagged as a
deliberate override of Day 3's original "small model is the right fit for
short transformation tasks" reasoning, not a reversal of it).

**Root cause of the network flakiness that shaped Day 6-7's decisions: not
actually a network block.** Systematically misdiagnosed for most of the
project as this dev machine's network intercepting HTTPS to `api.groq.com`
with a malformed cert (see the now-corrected note below). The real cause,
found only once the user paused it: **Avast's HTTPS-scanning feature** was
intercepting the TLS connection with its own certificate — trusted by
Windows (why `curl`/PowerShell always succeeded) but not by Python's
`certifi` store (why every Python call failed). Nothing wrong with Groq, the
network, or the code. Worth knowing if this resurfaces on a machine with
similar AV software.

**Document lifecycle — temporary vs. permanent delete (`vector_store.py`,
`main.py`, Streamlit Upload page).** `DELETE /api/v1/documents` previously
only hard-deleted. Added `set_active_by_source()` — a soft delete that flips
an `active` metadata flag (`$ne: False` in every read path, verified
empirically that this correctly treats chunks with no `active` key at all —
i.e. everything indexed before this feature existed — as active, no
migration needed) instead of removing the embedding, so restoring is
instant with no re-upload/re-embedding. `DELETE ...?permanent=true|false`
(default `true`, matching the original endpoint's exact behavior so the
existing UI button needed no changes) plus a new
`POST /api/v1/documents/restore`. `GET /api/v1/stats` gained
`hidden_sources`. UI: 🙈 hide / 🗑️ delete per document, restore from a
"Hidden documents" expander.

**Query expansion wired in (`app/graph/nodes.py`, `build_graph.py`).**
Written Day 3, flagged unwired in Day 4's log, still unwired until now. New
`expand` node sits between `rewrite` and `retrieve`
(`rewrite → expand → retrieve → fuse → ...`); `retrieve_node` now searches
the rewritten query *and* every expansion, RRF-merging each retriever's
own results across variants back into one list per retriever — reusing
`reciprocal_rank_fusion` for a second job (collapsing phrasings, not just
merging dense/BM25) since the mechanism only ever cares about rank
position. `fuse_node`/`rerank_node`/the retrieval-proof table needed zero
changes. Best-effort: `expand_node` falls back to `[]` on LLM failure
rather than failing the whole turn, unlike `rewrite_node`/`generate_node`
(no such cushion) — expansion is a recall booster, not a hard requirement.
Exposed end-to-end as `expanded_queries` (API response, Streamlit
expander).

**API error handling (`app/main.py`).** No global exception handler
existed — an unhandled exception (e.g. the Groq connection errors above)
fell through to Starlette's default bare `"Internal Server Error"`
**plain-text** response. The Streamlit client always calls
`response.json()` on a non-200, which raised its own unrelated
`JSONDecodeError` on a plain-text body, masking the real error. Fixed with
`@app.exception_handler(Exception)` returning `{"error": "internal_error",
"detail": str(exc)}` — every response is now guaranteed valid JSON;
full traceback still goes to the server log. Streamlit's error handling
also hardened defensively (try/except around its own `response.json()`
call) regardless.

**Image understanding — vision fallback for OCR-empty images
(`app/loaders/ocr_loader.py`, `llm_gateway.py`, `settings.py`).** Motivating
case: an X-ray upload failed outright — OCR correctly finds no text in a
photo that never had any, but the loader treated that as a hard error.
`load_image()` now falls back to `get_vision_completion()` (a real Groq
vision model call) when OCR extracts nothing, so a photo/scan with no text
in it (X-ray, diagram, equipment photo) still produces something
indexable. Chunk metadata records `extraction_method` ("ocr"/"vision") and
`vision_model` so which path ran is checkable, not assumed. Groq-only, no
Ollama fallback (deliberate — no local vision model installed, and
irrelevant for a hosted demo deployment that won't have Ollama available
at all); raises loudly if the vision call fails rather than silently
degrading.

Getting an actually-correct, working vision model took three real,
sequential bugs, each found only once Avast was paused and a live Groq
connection was actually possible:
1. **Wrong model ID.** Guessed `meta-llama/llama-4-scout-17b-16e-instruct`
   from Groq's own docs, which gave self-contradictory answers across
   repeated fetches (likely JS-rendered content the fetcher couldn't read).
   Verified against a live `client.models.list()` call: that model doesn't
   exist on this account at all. `qwen/qwen3.6-27b` is the only real
   vision-capable model available — `settings.groq_vision_model`, no
   fallback list (nothing else to fall back to).
2. **Unusable output from a "thinking" model.** Unfixed, `qwen3.6-27b`
   dumped 500+ words of rambling chain-of-thought reasoning about a test
   X-ray and got cut off by the token limit before ever stating an answer
   — useless as an indexed chunk either way. Fixed with
   `reasoning_effort="none"` + `max_completion_tokens=600` — verified live,
   clean 3-5 sentence factual output on the first try.
3. **Confidently wrong identity claims.** The vision prompt originally let
   the model name specific real people it recognized (or thought it did) —
   a real upload came back describing a well-known actress's photo as a
   *different*, wrong actress by name, stored as if it were fact. Real
   models are unreliable at face recognition; a wrong name is worse than no
   name, since it gets indexed and answers questions about the wrong
   person entirely. `_VISION_PROMPT` now explicitly forbids asserting a
   person's identity unless a name is legibly written/captioned in the
   image itself — describes visible appearance/context instead.

**Two more real bugs found through the same live image testing, both
fixed:**
- **Filename baked in wrong.** `load_image()` only ever saw the temp
  UUID-prefixed path the upload route saves to (`8fda1f12-....jpg`) — the
  original filename gets patched into `metadata["source"]` afterward in
  `main.py`, but that's too late for text already written into
  `page_content`. Fixed by giving `load_image()` an explicit
  `original_filename` parameter, used consistently for both the metadata
  and the `"Filename: ..."` line prepended to vision-fallback content
  (added specifically so a filename-based hint like "explain the
  sai_pallavi image" has something to match, given point 3 above means the
  description itself may not name anyone).
- **Cross-document content blending in `generate_node`.** `context_text`
  concatenated every retrieved chunk's raw `page_content` with no
  indication of which source document it came from. Caught for real: two
  unrelated images retrieved for one question got blended into a single
  answer that attributed one image's visual details to the other, and
  fabricated a detail ("a bird") supported by neither individually.
  `generate_node` now labels every chunk `[Source: <filename>]` in the
  prompt, and the system prompt explicitly forbids blending facts across
  source blocks. Not image-specific — any two chunks from different
  documents were equally at risk; images just made a wrong answer visible
  immediately.

**`generate_node`'s refusal behavior split (greeting vs. genuinely
off-topic).** Previously a plain "hi" got the exact same flat refusal
sentence as "write me a Python function" — technically consistent with
Day 7's anti-hedging fix, but read as broken. System prompt now carves out
a narrow exception: a message that is *purely* greeting/small talk (no
real question in it) gets a short warm reply instead. Anything with actual
content — including off-topic content — still goes through the identical
strict "answer only from context, or refuse verbatim" rule as before, so
this doesn't reopen the hedge-then-answer door Day 7 closed.

**Tesseract-OCR installed locally (this machine only).** Was never
installed on this Windows box at all — `TESSERACT_CMD` pointed at a Linux
path (`/usr/bin/tesseract`, still correct for the Docker deployment
target, left untouched in `.env.example`). Installed via
`winget install --id UB-Mannheim.TesseractOCR -e --source winget` (the
default `msstore` source hit the same Avast interception as Groq;
`--source winget` bypassed it); `.env`'s `TESSERACT_CMD` updated to the
real Windows path for local dev only.

**Known, unresolved: `<candidate>_Resume.pdf` went missing from the
corpus and was never recovered.** Traced (not fully proven) to concurrent
multi-process access to the local Chroma persistent store — at one point,
5 separate processes (repeated restarts across the session, some of which
Windows' own tools reported as killed while they kept answering on their
port regardless — never fully explained) held simultaneous connections to
the same `data/chroma_db` directory, which Chroma's local mode isn't
designed to support safely. User's call: re-upload it manually rather than
keep investigating. Worth remembering if corpus data looks inconsistent
again — check for multiple live processes against the same persist dir
before assuming a code bug.

**Confirmed, not yet built: separate downstream copilot integration.**
Discussed and confirmed as architecturally correct, no code changes needed:
a separate project (e.g. Engineering Copilot) in its own repo, its own
deployment, its own multi-agent logic, consuming UniRAG purely through its
existing REST API (`/upload`, `/chat`, `/search`) — exactly the
"AI Knowledge Core with separate downstream consumers" shape CLAUDE.md §1
describes. One real gap flagged for whenever this actually happens: no
per-project data isolation exists today — every upload goes into one shared
corpus (visibly true after tonight's testing left resumes, X-rays, and test
images all coexisting). Two options discussed, neither built: a second,
fully separate UniRAG deployment per consumer (simplest, zero code), or
wiring the already-built-but-unused `app/filtering/metadata_filter.py` into
upload/retrieval so one shared instance can tag and filter by project.
Decision deferred — not needed until a second project actually starts
uploading.

### Day 9 — Pre-push cleanup for the first public commit (done)

Not a build day — a pass over the whole repo before the first push to
GitHub, done in chat with Claude Code rather than as a coding session.

**§2's "do not use GitHub Actions" constraint is reversed, on request.**
`.github/workflows/ci.yml` (a plain pytest-smoke-suite-on-push/PR workflow,
already written, no deploy step — Render deploys natively from `render.yaml`
instead) existed in the working tree with no matching approved-deviation
entry in this log, so it got flagged during the pre-push review. Asked
about directly; the answer was to keep CI, not drop it — it's free on a
public repo and a real signal for anyone reviewing the project. §2 updated
to drop the "GitHub Actions" line rather than leave a stale constraint next
to a workflow file that violates it.

**`CLAUDE.md` itself relocated to `docs/CLAUDE.md`**, on request, to get it
out of the repo root. All six references to it in `README.md` (the
"map/territory" line, the project-structure tree, and four inline
citations) were updated to the new path in the same pass — nothing links to
a dead file.

**One real PII fix, not just a deviation:** the Day 8 entry above named a
third party by their real full name via a resume filename
(`<candidate>_Resume.pdf` is the redacted form now in this file — it
originally had the actual name). Caught during the same pre-push scan,
redacted on request before anything was pushed. Everything else scanned
clean — no API keys, no other personal data, `.env`/`venv/`/`data/` all
correctly gitignored.

### Environment notes (not code, but cost real time — worth knowing)

- Project venv lives at `D:\projects\UniRag\venv` (Python 3.12.7), not the
  `python` on PATH (which resolves to an unrelated Python 3.14 install with
  none of `requirements.txt` present). Always run via
  `./venv/Scripts/python.exe` or activate the venv first.
- **Corrected on Day 8, was wrong for most of the project**: this was never
  a real network/certificate block. It was **Avast's HTTPS-scanning
  feature** intercepting the TLS connection to `huggingface.co`/
  `api.groq.com` with its own certificate — trusted by Windows tools
  (`curl`, PowerShell), not by Python's `certifi` store, which is why every
  Python call failed while everything else looked fine. Pausing Avast
  fixes it immediately; no cert/network fix was ever actually needed.
  Hugging Face models still load fine offline either way
  (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, downloaded once via
  `hf download <repo-id>`). The Groq path (`llm_gateway.py`, vision
  included) is now verified working end-to-end for real, live, with Avast
  paused — see Day 8's log.
- Tesseract-OCR is now installed locally too (Day 8), via
  `winget install --id UB-Mannheim.TesseractOCR -e --source winget` — the
  default `msstore` source hits the same Avast interception, `--source
  winget` bypasses it. `.env`'s `TESSERACT_CMD` points at the real Windows
  path for local dev; `.env.example` correctly still says
  `/usr/bin/tesseract` for the Docker/Linux deploy target.
- `.gitignore`, `README.md`, and `tests/test_loaders.py` /
  `test_retrieval.py` / `test_api.py` are **no longer empty** — see the
  correction in "Still stubbed" below; this note was stale.

### Still stubbed / not started

All 6 required days plus Day 7's UI are done, plus Day 8's live-testing
hardening pass (LLM resilience, document lifecycle, image understanding —
see that section above). Corrected from an earlier stale version of this
list: `.gitignore`, `README.md`, and all three `tests/test_*.py` files
were previously logged here as empty/missing — they are not; substantive
content exists in all of them (not tracked to a specific day's log entry,
so it's not detailed above, but real and current). What's actually left:

- **Render deploy** (Day 7, optional) — not attempted; needs the user's own
  account/credentials, see Day 7 log above for why.
- `/chat` doesn't stream (returns the full answer in one response) —
  CLAUDE.md's API surface doesn't call for streaming, so this wasn't built;
  would be a `StreamingResponse` change to the one route if wanted later.
- Long-term conversation memory (persistent storage past one process's
  lifetime) is still explicitly stubbed with a `TODO` in
  `conversation_memory.py`, per the tech stack table. `session_memory.py`
  exists (Day 4) but still isn't wired into any route.
- The Domain selector in the Streamlit sidebar is visual-only (confirmed
  choice, see Day 7 log) — doesn't filter retrieval. Directly relevant now
  that Day 8 confirmed the plan to integrate separate downstream copilots:
  this selector (or `app/filtering/metadata_filter.py`, already built but
  unused) is the natural place to wire in real per-project data isolation
  whenever that's actually needed — see Day 8's log for the two options
  discussed.
- No per-project/per-tenant data isolation exists — one shared corpus for
  everything uploaded (Day 8, flagged for whenever a second downstream
  consumer, e.g. an Engineering Copilot, actually starts uploading its own
  documents).
