# UniRAG — Project Summary

A reusable, domain-agnostic **AI Knowledge Core**: a production-shaped
hybrid-retrieval RAG platform, not a single-purpose chatbot. No
domain-specific logic in `app/`; everything is config/metadata-driven.
Optimized for legibility: every non-obvious choice has an inline "why" note.
Full day-by-day log lives in `docs/CLAUDE.md` §12; this file is the
fast-recall version.

## Tech stack
| Layer | Choice |
|---|---|
| Backend | FastAPI + Uvicorn (sync routes — see Gotchas) |
| AI framework | LangChain + LangGraph |
| Loaders | LangChain loaders; `pytesseract` OCR; Groq vision fallback for text-less images |
| Chunking | Recursive + Semantic |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local) |
| Vector DB | ChromaDB (persistent, `./data/chroma_db`) |
| Retrieval | BM25 + dense → RRF fusion → metadata filter → context compression → cross-encoder rerank |
| Query enhancement | LLM rewrite + expansion (wired into the graph) |
| LLM gateway | Groq (multi-model fallback list) → auto-falls back to Ollama on total Groq failure |
| Guardrails | NeMo Guardrails, regex rails only (PII + jailbreak patterns), no LLM self-check |
| Memory | Session + conversation (in-process only; long-term persistence stubbed) |
| Config | Pydantic Settings + `.env` |
| Tests | Pytest smoke tests only |
| Deploy | Docker + Docker Compose; Render Blueprint prepared, not deployed |
| UI | Streamlit, thin client — talks to the API only, no shared imports |
| CI | GitHub Actions — tests only, no deploy step |

## Architecture
Request flow (`/api/v1/chat`): `input_guard → graph.run_turn() → output_guard`
Graph (LangGraph, linear): `rewrite → expand → retrieve → fuse → rerank → compress → generate → save_turn`
- `VectorStore` (Chroma) is the single source of truth for the corpus; `BM25Retriever` is built from it, so both retrievers rank identical chunks keyed by shared `chunk_id`.
- `reciprocal_rank_fusion` merges by **rank position**, not raw score — used twice: dense-vs-BM25 fusion, and collapsing multiple query-expansion phrasings.
- `generate_node` answers from the *original* question + compressed, per-source-labeled context; refuses verbatim on off-topic (no hedge-then-answer), replies warmly to pure greetings.

## API surface
`POST /api/v1/upload`, `POST /api/v1/chat`, `POST /api/v1/search` (retrieval-only, no LLM), `GET /api/v1/health`, `GET /api/v1/stats`, `DELETE /api/v1/documents` (`?permanent=true|false`), `POST /api/v1/documents/restore`.

## Status: all 6 required days + Day 7 UI + Day 8 hardening — done, verified live (not just claimed)
- **Days 1–7**: loaders/OCR/chunking → embeddings/Chroma/BM25/RRF → rewrite/expansion/filter/rerank → compression/LangGraph/memory → guardrails/eval/logging → FastAPI/Docker → Streamlit UI. Each finish line was run and observed, not assumed.
- **Day 8 (post-launch hardening, from live testing)**:
  - LLM gateway: Groq multi-model fallback, then auto-fallback to Ollama (was manual restart before); answering model/provider now returned in the API response.
  - Document lifecycle: soft delete (`active` flag) vs. permanent delete, plus restore.
  - Query expansion wired into the graph (was written Day 3, unused until now).
  - Global exception handler — every error response is guaranteed valid JSON.
  - Image understanding: OCR-empty images fall back to Groq vision (`qwen/qwen3.6-27b`), with `reasoning_effort="none"` to avoid rambling, and an explicit ban on asserting a person's identity (real models guess wrong; a wrong name is worse than none).
  - Cross-document blending fixed: `generate_node` now labels every chunk `[Source: <filename>]` and is told not to mix facts across sources.

## Known limitations (not bugs — deliberate/tracked)
- No per-tenant data isolation — one shared corpus for everything uploaded so far.
- Long-term conversation memory is a stubbed `TODO`; resets on process restart.
- `/chat` doesn't stream (one blocking call).
- Streamlit's "Domain" selector is visual-only, doesn't filter retrieval.
- Render deploy not attempted (needs the user's own account/credentials).
- One document went missing from the corpus during testing, suspected cause: multiple processes hitting the same local Chroma dir concurrently (Chroma's local mode isn't safe for that) — never fully proven, re-uploaded manually.

## Gotchas worth remembering
- Routes in `main.py` are sync `def`, not `async def` — guardrails call `asyncio.run()` internally, which errors inside an already-running event loop.
- The "network block" that shaped early workarounds (HF/Groq HTTPS failing in Python only) was **Avast's HTTPS scanning**, not a real network/cert issue — pausing Avast fixes it instantly.
- Run everything via the project's own `venv` (`venv\Scripts\python.exe` on Windows, `venv/bin/python` on Linux/Mac) — a `python` already on `PATH` may point at an unrelated install.

## Module deep-dives (each has a "Why this works" note + sample Q&A in its own docstring)
`hybrid_fusion.py` (RRF) · `rewrite.py`/`expansion.py` (prompt design) · `cross_encoder_rerank.py` · `build_graph.py` (LangGraph, has ASCII diagram) · `input_guard.py`/`output_guard.py` (deterministic vs. LLM self-check tradeoff).
