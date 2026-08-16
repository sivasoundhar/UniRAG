# UniRAG UI Upgrade Spec

## Goal
This is a public-facing demo, shared as a link. The goal isn't a
feature-complete app — it's making the sophistication of the backend
(hybrid retrieval, RRF fusion, reranking, guardrails) visible to a stranger
within 30 seconds of opening the link, with zero setup required on their
end.

## Decision: Custom CSS, not Tailwind
Considered both. Going with **custom CSS** (CSS variables injected via
`st.markdown`), not Tailwind CDN, because:
- This is a single-page Streamlit demo, not a multi-component framework —
  Tailwind's real value (utility-class reuse across many components) doesn't
  apply here
- Adds CDN load weight for no real benefit in this context
- Custom CSS variables (`--accent`, `--panel`, etc.) are shorter and more
  readable for one page than Tailwind's utility classes would be
- This project's signal is about RAG technique, not frontend framework
  choice — Tailwind wouldn't add credibility here
- Revisit Tailwind only if this ever becomes a real multi-page React
  frontend later (not in current scope)

## Design Language
- **Theme:** dark, technical/lab aesthetic — not a generic SaaS chatbot look
- **Fonts:** `Space Grotesk` (headings), `Inter` (body), `IBM Plex Mono`
  (data/pipeline readouts — reinforces "this is a system, not just a chat box")
- **Palette:**
  | Token | Hex | Use |
  |---|---|---|
  | `--bg` | `#0D1117` | page background |
  | `--panel` | `#141A22` | cards/sidebar |
  | `--panel-2` | `#1A222D` | nested elements, user bubble |
  | `--line` | `#262F3D` | borders |
  | `--text` | `#E9ECF1` | primary text |
  | `--muted` | `#6E7787` | secondary text |
  | `--accent` | `#4FD1C5` | teal — active/success signal |
  | `--accent-dim` | `#23433F` | accent borders |
  | `--amber` | `#F0B429` | highlight (fused rank, sources marker) |

## Components to Add

### 1. "How this works" panel (highest priority)
Sits above the chat, always visible. Shows the pipeline as 5 icon-labeled
steps: rewrite → hybrid retrieve → rerank → compress → guardrails, each
with a one-line description. Purpose: a stranger understands in 3 seconds
that this isn't a plain LLM wrapper.
- Use `st.container(border=True)` + `st.columns`, not custom divs — stay
  consistent with the rest of `app.py`'s native-component style.

### 2. Retrieval proof table (highest priority)
Under each assistant answer, a small table: chunk name, BM25 rank, dense
rank, fused (RRF) rank, final reranked position. This is the strongest
differentiator — visibly proves hybrid retrieval + RRF + reranking are real,
not just claimed.
- **Backend change required:** `/api/v1/chat` response needs a
  `retrieval_proof` field:
  ```json
  [{"chunk": "...", "bm25_rank": 1, "dense_rank": 3, "fused_rank": 1, "rerank_position": 1}]
  ```
  `app/retrieval/hybrid_fusion.py` currently returns only the final fused
  list — extend it to expose per-chunk ranks before generation, then thread
  that through the API response.
- Render with `st.dataframe` or `st.table`, not custom HTML — match the
  rest of the file's native-component approach.

### 3. Pre-loaded corpus (removes setup friction)
Load 2-3 sample documents server-side on startup so Chat/Search work
immediately when someone clicks the link, without requiring them to upload
anything first. Upload stays available as a secondary action.

### 4. Visual polish pass (CSS only, no logic changes)
Update the existing `.unirag-header`, `.stage`, `.meta-row` styles in
`app.py` to the palette above. Keep the file's current light/dark theme
compatibility intent. Do not remove the honesty comment about stages
showing "all done" after the fact (since `/chat` isn't streaming).

### 5. Domain toggle — leave as-is
Already correctly labeled "Preview only — doesn't filter retrieval yet."
Don't wire it to real filtering and don't remove it — the honest labeling
is good practice, not a gap to fix for this demo.

## Explicitly Out of Scope for This Upgrade
- Fake real-time pipeline animation (dishonest — `/chat` is a single
  blocking call, not a stream). Still out of scope specifically: lighting
  up stages *sequentially* as if each one's real progress were known. What
  Revision 2 *did* add (see below) is different in kind, not a walk-back of
  this rule: all stages pulsing together while the request is in flight,
  asserting only "the pipeline is running," never fabricated per-stage timing.
- Making the domain toggle functionally filter retrieval
- Any Tailwind/CSS framework migration
- Multi-page dashboard features (Analytics, Settings, Help, account/upsell
  UI) — SaaS-product theater, doesn't demonstrate RAG technique
- Changing `app.py`'s architecture — it stays a pure HTTP client with zero
  RAG logic, per docs/CLAUDE.md

## Reference Mockups
Two HTML concept files were built during design discussion (not for direct
use — Streamlit can't render them as-is, they're visual references only):
- `unirag_ui_v3_demo.html` — custom CSS version (chosen direction)
- `unirag_ui_v4_tailwind.html` — Tailwind version (rejected, kept as future
  reference only if this ever becomes a React frontend)

---

## Revision 2 (implemented) — matched directly to unirag_ui_v3_demo.html

After the first pass (native `st.container`/`st.columns`/`st.dataframe` per
the decisions above) the user supplied a screenshot and then the actual
`unirag_ui_v3_demo.html` source, asking for pixel-level fidelity to it. That
changed several decisions above — logged here rather than silently, per
docs/CLAUDE.md's "say out loud you're deviating" rule.

**What changed from the original plan:**
- **#1 "How this works" panel and #2 retrieval-proof table are now custom
  HTML/CSS**, not native `st.container`/`st.columns`/`st.dataframe`. The
  mockup's connected-icon-box diagram and badge-highlighted table aren't
  achievable with those native components. Everything else (sidebar nav,
  file uploader, chat input, sliders) stayed native Streamlit, restyled by
  palette/theme rather than by fighting Streamlit's internal DOM — except
  where the mockup's nav look (plain muted text, only the active item
  boxed) genuinely required overriding `[data-testid="stSidebar"] button`
  styles; that's scoped narrowly to the sidebar and documented inline as a
  structural-but-unstable-class-name risk.
- **Icons are monochrome glyphs** (`✎ ⇄ ↕ ▤ ✓`), not emoji — matches the
  mockup's single-color "technical/lab" look; colorful emoji broke it.
- **#5 Domain toggle: removed entirely**, on direct request — supersedes
  the original "leave as-is, don't remove it." The Documents/Chunks metric
  count survived instead of being deleted, tucked into a collapsed
  "⚙️ Corpus stats" expander so the sidebar's default view matches the
  mockup's clean nav → pre-loaded-corpus → footer-stats shape.
- **`/api/v1/stats` gained a `sources` field** (list of indexed filenames)
  so the sidebar's "Pre-loaded corpus" list shows what's actually indexed,
  not a hardcoded copy of the mockup's illustrative example filenames.
- **New: a live "pipeline running" highlight while the blocking `/chat`
  call is in flight** — every stage pill pulses together (CSS animation)
  from the moment the request is sent until the real response replaces it.
  This is *not* the sequential fake-per-stage animation the original scope
  explicitly ruled out below — it asserts one true fact ("the pipeline is
  executing right now"), never a false one ("step 3 of 6 is running"),
  since `/chat` has no per-stage progress to report honestly.
- **Retrieval-proof highlighting logic**, reverse-engineered from the
  mockup's actual markup rather than guessed: BM25/dense columns badge only
  where that value is `1` (this retriever's own top pick); the Fused (RRF)
  column is always amber-badged (it's the mechanism being demonstrated);
  the Reranked column is badged only when the chunk was *also* some
  retriever's own `#1` — a real, computed claim ("this final result was
  independently corroborated"), not a blanket always-on rule.

Everything else in this spec (design language, palette, custom-CSS-not-
Tailwind decision, pre-loaded corpus, out-of-scope list) still holds.
