"""
Thin Streamlit UI for UniRAG — a pure HTTP client of app/main.py's API.

"Thin" means what it says: this file has no RAG logic of its own and never
imports from app/. Every action (chat, upload, search) is a `requests` call
to the FastAPI service; this file only renders what comes back. That's also
why it's a separate deployable unit with its own requirements.txt/Dockerfile
(see docker-compose.yml) rather than living inside the app/ package.

Deliberately domain-neutral (no "medical", no product branding) — UniRAG is
the reusable core; a downstream copilot (per CLAUDE.md's architecture) would
skin this or build its own UI on top of the same API, not the other way
around.

Visual design follows UI_UPGRADE_SPEC.md's second revision (the v3 mockup,
built to match a specific reference screenshot the user supplied). That's a
deliberate, discussed deviation from the spec's first revision, which kept
custom HTML to the header/pipeline-pills/meta-row only and used native
st.container/st.dataframe for everything else: matching the mockup's icon
pipeline diagram and badge-highlighted retrieval-proof table isn't
achievable with those native components, so this file uses custom HTML/CSS
for those two pieces specifically. Sidebar navigation, inputs, and buttons
stay native Streamlit widgets — restyled by palette/theme, not by fighting
Streamlit's internal DOM, since that structure isn't a stable target across
versions.
"""

import html
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="UniRAG", page_icon="📚", layout="wide")

# Custom CSS for the pieces native Streamlit components can't render:
# the header, the pipeline diagram ("How this works" + per-answer stage
# pills), the chat bubbles, and the retrieval-proof table. The base dark
# theme itself lives in .streamlit/config.toml (a real Streamlit theme, not
# a CSS override) — every color below references that theme's palette via
# CSS variables so the hand-built pieces sit on top of it consistently
# rather than assuming a background that may not be there.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0D1117;
        --panel: #141A22;
        --panel-2: #1A222D;
        --line: #262F3D;
        --text: #E9ECF1;
        --muted: #6E7787;
        --accent: #4FD1C5;
        --accent-dim: #23433F;
        --amber: #F0B429;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }

    /* ---- Header ---- */
    .unirag-header {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent-dim) 0%, var(--panel-2) 100%);
        border: 1px solid var(--line);
        color: var(--text);
        margin-bottom: 1.5rem;
    }
    .unirag-header h1 { margin: 0; font-size: 1.6rem; font-family: 'Space Grotesk', sans-serif; color: var(--text); }
    .unirag-header p { margin: 0.25rem 0 0 0; color: var(--muted); }

    /* ---- "How this works" pipeline diagram ---- */
    .unirag-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1.1rem 1.4rem 1.3rem 1.4rem;
        margin-bottom: 1.25rem;
    }
    .how-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--text); }
    .how-subtitle { color: var(--muted); font-size: 0.83rem; margin: 0.15rem 0 1.1rem 0; }
    .pipeline-diagram { display: flex; align-items: flex-start; }
    .pd-step { display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center; flex: 1; min-width: 90px; }
    .pd-icon {
        width: 34px; height: 34px; border-radius: 9px; border: 1px solid var(--accent-dim);
        background: var(--panel-2); color: var(--accent);
        display: flex; align-items: center; justify-content: center; font-size: 15px;
    }
    .pd-label { font-weight: 500; font-size: 0.72rem; color: var(--text); font-family: 'IBM Plex Mono', monospace; }
    .pd-caption { font-size: 0.66rem; color: var(--muted); max-width: 100px; line-height: 1.3; }
    .pd-arrow { color: var(--line); font-size: 0.95rem; margin: 17px 2px 0 2px; }

    /* ---- Per-answer stage pills ---- */
    .pipeline { display: flex; align-items: center; gap: 5px; margin: 2px 0 10px 0; flex-wrap: wrap; }
    .stage {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 3px 9px;
        border-radius: 5px; border: 1px solid var(--accent-dim);
        color: var(--accent); background: rgba(79, 209, 197, 0.10);
    }
    .stage-arrow { color: var(--muted); font-size: 11px; }
    /* Shown only while a request is in flight (see send_chat_message) — all
       stages pulse together, never one at a time. /api/v1/chat is a single
       blocking call with no per-stage progress to report, so lighting up
       stages sequentially would be fabricating timing data that doesn't
       exist. Pulsing together instead signals a true fact ("the pipeline
       is executing right now") without claiming a false one. */
    .stage.pending { animation: unirag-pulse 1.3s ease-in-out infinite; }
    @keyframes unirag-pulse {
        0%, 100% { opacity: 0.5; box-shadow: 0 0 0 rgba(79, 209, 197, 0); }
        50% { opacity: 1; box-shadow: 0 0 8px rgba(79, 209, 197, 0.45); }
    }

    .meta-row {
        display: flex; gap: 16px; margin: 4px 0 8px 0; flex-wrap: wrap;
        font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--muted);
    }
    .meta-row b { color: var(--amber); font-weight: 600; }

    /* ---- Chat bubbles ---- */
    .chat-row { display: flex; margin: 0.7rem 0; }
    .chat-row.user { justify-content: flex-end; }
    .user-bubble {
        background: var(--panel-2); border: 1px solid var(--line);
        border-radius: 12px 12px 3px 12px; /* sharp bottom-right corner reads as a speech-bubble tail */
        padding: 0.7rem 0.95rem; max-width: 70%; color: var(--text); font-size: 0.88rem;
    }
    .answer-text {
        border-left: 2px solid var(--accent); padding-left: 0.95rem;
        margin: 0 0 0.9rem 0; font-size: 0.9rem; color: var(--text); line-height: 1.55;
    }

    /* ---- Retrieval-proof table ---- */
    .proof-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
    .proof-table th {
        text-align: left; color: var(--muted); font-weight: 500; font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 0.03em; font-family: 'IBM Plex Mono', monospace;
        padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--line);
    }
    .proof-table td { padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--line); color: var(--text); }
    .proof-table tr:last-child td { border-bottom: none; }
    .proof-table td:first-child { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; }
    /* Every rank is a pill (muted by default) — .top/.fused recolor it, they
       never remove it — matching the reference's "always a badge" rule
       rather than plain unstyled numbers, which drew the eye unevenly. */
    .rank-badge {
        display: inline-block; padding: 0.1rem 0.45rem; border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
        background: var(--panel-2); color: var(--muted);
    }
    .rank-badge.top { color: var(--accent); border: 1px solid var(--accent-dim); }
    .rank-badge.fused { background: rgba(240, 180, 41, 0.12); color: var(--amber); font-weight: 500; }
    .rank-dash { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; }

    /* ---- Sidebar logo + status ---- */
    .sidebar-logo { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.9rem; }
    .logo-badge {
        width: 32px; height: 32px; border-radius: 8px;
        background: linear-gradient(145deg, var(--accent), var(--accent-dim));
        color: #0A0F14; display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.75rem; flex-shrink: 0;
    }
    .logo-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--text); line-height: 1.1; }
    .logo-status { font-size: 0.7rem; color: var(--muted); margin-top: 0.2rem; font-family: 'IBM Plex Mono', monospace; }
    .status-dot { display: inline-block; width: 5px; height: 5px; border-radius: 50%; margin-right: 5px; }
    .status-dot.online { background: var(--accent); }
    .status-dot.offline { background: #E5484D; }

    /* ---- Sidebar nav: plain muted rows, only the active one gets a
       subtle boxed highlight — the reference has no filled "button" look
       at all for nav, so secondary buttons are stripped down to bare text
       and only the primary (active) one gets a faint panel tint. Scoped to
       the sidebar so the real action buttons elsewhere (Search, Index this
       document) keep their normal solid CTA styling. */
    [data-testid="stSidebar"] button {
        justify-content: flex-start !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 0.6rem !important;
        box-shadow: none !important;
    }
    /* Streamlit wraps the button label in its own inner flex div that
       re-centers regardless of the button's own justify-content — found by
       inspecting the live DOM (st-emotion-cache-*, unstable class names but
       a stable *structural* pattern: "the div directly inside a button").
       Overriding that div, not just the button, is what actually
       left-aligns the label. */
    [data-testid="stSidebar"] button > div {
        justify-content: flex-start !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] button p { text-align: left !important; }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: var(--panel-2) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: var(--panel-2) !important;
        border-color: var(--accent-dim) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# The real stages app/graph/build_graph.py runs, in order — used both for
# the always-visible "How this works" diagram and the per-answer pills.
# "expand" added alongside the app/query/expansion.py wiring — see
# render_expanded_queries below for where its actual output (not just its
# name in this list) becomes visible per answer.
PIPELINE_STAGES = ["rewrite", "expand", "retrieve", "fuse", "rerank", "compress", "generate"]

# Monochrome glyphs (not colorful emoji) so the icon boxes stay teal-tinted
# and consistent with the rest of the "technical/lab" palette, matching
# unirag_ui_v3_demo.html's reference icon set rather than looking like a
# generic emoji-decorated app.
HOW_IT_WORKS_STEPS = [
    ("✎", "rewrite", "clean up raw query"),
    ("⎇", "expand", "LLM generates alternate phrasings"),
    ("⇄", "hybrid retrieve", "BM25 + dense, RRF fused"),
    ("↕", "rerank", "cross-encoder reorder"),
    ("▤", "compress", "trim to relevant context"),
    ("✓", "guardrails", "PII / jailbreak check"),
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "uploads" not in st.session_state:
    st.session_state.uploads = []
if "page" not in st.session_state:
    st.session_state.page = "Chat"


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    api_online = False
    try:
        requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5).raise_for_status()
        api_online = True
    except requests.RequestException:
        pass

    status_class = "online" if api_online else "offline"
    status_text = "core online · demo mode" if api_online else f"core offline · can't reach {API_BASE_URL}"
    st.markdown(
        '<div class="sidebar-logo">'
        '<div class="logo-badge">UR</div>'
        '<div><div class="logo-title">UniRAG</div>'
        f'<div class="logo-status"><span class="status-dot {status_class}"></span>{status_text}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.caption("NAVIGATE")
    # Native st.button per nav row rather than a segmented control, so the
    # active page can be shown Streamlit-natively (type="primary") instead
    # of hand-styling a widget's internal DOM — the CSS above then strips
    # that primary/secondary pair down to the reference's subtle look
    # (plain muted text vs. a faint boxed highlight, not a filled CTA).
    for icon, label in [("💬", "Chat"), ("🔍", "Search & Retrieval"), ("📤", "Upload")]:
        is_active = st.session_state.page == label
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    if st.button("🆕  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"conversation: `{st.session_state.conversation_id[:8]}…`")

    st.divider()
    corpus_sources: list[str] = []
    corpus_stats: dict = {}
    if api_online:
        try:
            corpus_stats = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5).json()
            corpus_sources = corpus_stats.get("sources", [])
        except requests.RequestException:
            pass

    st.caption("PRE-LOADED CORPUS")
    for source in corpus_sources[:8]:
        st.markdown(
            f"<div style='font-size:0.78rem;color:var(--muted);font-family:\"IBM Plex Mono\",monospace;line-height:1.9;'>"
            f"<span style='color:var(--accent);'>●</span> {html.escape(source)}</div>",
            unsafe_allow_html=True,
        )
    if len(corpus_sources) > 8:
        st.caption(f"+ {len(corpus_sources) - 8} more")

    # Live document/chunk counts are real info the reference doesn't show
    # in its default sidebar view — tucked into a collapsed section instead
    # of deleted outright, so it's still one click away without competing
    # with the clean nav+corpus look. The domain toggle (spec's original
    # #5) is gone entirely, not just collapsed — removed on direct request,
    # a deliberate deviation from UI_UPGRADE_SPEC.md's "don't remove it".
    if corpus_stats:
        with st.expander("⚙️ Corpus stats", expanded=False):
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Documents", corpus_stats.get("document_count", 0))
            stat_col2.metric("Chunks", corpus_stats.get("chunk_count", 0))

    st.markdown(
        '<div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid var(--line);'
        'font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:var(--muted);line-height:2;">'
        '<div style="display:flex;justify-content:space-between;"><span>retrieval</span><span style="color:var(--accent);">hybrid+RRF</span></div>'
        '<div style="display:flex;justify-content:space-between;"><span>rerank</span><span style="color:var(--accent);">cross-encoder</span></div>'
        '<div style="display:flex;justify-content:space-between;"><span>llm</span><span style="color:var(--accent);">groq/ollama</span></div>'
        '<div style="display:flex;justify-content:space-between;"><span>guardrails</span><span style="color:var(--accent);">active</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


# --- Header --------------------------------------------------------------

st.markdown(
    '<div class="unirag-header"><h1>📚 UniRAG</h1>'
    "<p>Ask questions grounded in your own documents — hybrid retrieval, "
    "reranking, and cited sources.</p></div>",
    unsafe_allow_html=True,
)

if not api_online:
    st.warning(
        f"Can't reach the UniRAG API at `{API_BASE_URL}`. "
        "Start it (`uvicorn app.main:app`) or check API_BASE_URL, then refresh."
    )


# --- Chat ----------------------------------------------------------------


def render_pipeline_explainer() -> None:
    """
    Always-visible panel above the chat: what actually happens to a query
    before an answer comes back — icon-and-arrow diagram, matching the
    reference mockup. Custom HTML (not st.container/st.columns) because the
    connected-icon-box layout isn't achievable with native components; see
    this file's module docstring for why that's a deliberate deviation from
    UI_UPGRADE_SPEC.md's first revision. Purpose stays the same: a stranger
    understands in ~3 seconds that this isn't a plain LLM wrapper.
    """
    steps_html = ""
    for i, (icon, label, caption) in enumerate(HOW_IT_WORKS_STEPS):
        if i > 0:
            steps_html += '<div class="pd-arrow">→</div>'
        steps_html += (
            '<div class="pd-step">'
            f'<div class="pd-icon">{icon}</div>'
            f'<div class="pd-label">{label}</div>'
            f'<div class="pd-caption">{caption}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="unirag-card">'
        '<div class="how-title">How this works</div>'
        '<div class="how-subtitle">Every answer below runs through this pipeline — nothing is a plain LLM call</div>'
        f'<div class="pipeline-diagram">{steps_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _rank_cell(value: int | None, *, highlight: bool = False, fused: bool = False) -> str:
    """
    Takes: a rank (or None if that ranker never surfaced the chunk at all)
    and whether this particular cell should be called out.
    Returns: the HTML for one <td>'s content. Every present rank renders as
    a pill — muted by default, matching the reference's "always a badge"
    style rather than bare numbers, which drew the eye unevenly across the
    row. A missing rank (BM25 or dense genuinely never returned this chunk)
    renders as a plain dash, not a badge, since there's no rank to badge.
    """
    if value is None:
        return '<span class="rank-dash">—</span>'
    classes = "rank-badge"
    if fused:
        classes += " fused"
    elif highlight:
        classes += " top"
    return f'<span class="{classes}">#{value}</span>'


def render_retrieval_proof(rows: list[dict]) -> None:
    """
    Takes: the retrieval_proof rows from a chat response (chunk, bm25_rank,
    dense_rank, fused_rank, rerank_position).
    Returns: nothing — renders a table (custom HTML inside a native
    st.expander) showing each stage's rank per chunk. Highlighting rules,
    matched to unirag_ui_v3_demo.html's actual reference markup rather than
    guessed:
      - BM25 / dense columns: highlighted only where that value is 1 —
        "this chunk was this retriever's own top pick."
      - Fused (RRF) column: always amber — it's the mechanism actually
        being demonstrated, so it's called out regardless of the number.
      - Reranked column: highlighted only when this chunk was ALSO some
        retriever's own #1 pick (bm25_rank == 1 or dense_rank == 1) — a
        real, computed claim ("this final result was independently
        validated by at least one retriever"), not a blanket "always on"
        rule, which would have implied every final result is equally
        well-supported when some are only there because fusion+reranking
        found them, not because either retriever alone did.
    """
    body_rows = ""
    for row in rows:
        chunk_label = html.escape(row["chunk"])
        corroborated = row["bm25_rank"] == 1 or row["dense_rank"] == 1
        body_rows += (
            "<tr>"
            f"<td>{chunk_label}</td>"
            f"<td>{_rank_cell(row['bm25_rank'], highlight=row['bm25_rank'] == 1)}</td>"
            f"<td>{_rank_cell(row['dense_rank'], highlight=row['dense_rank'] == 1)}</td>"
            f"<td>{_rank_cell(row['fused_rank'], fused=True)}</td>"
            f"<td>{_rank_cell(row['rerank_position'], highlight=corroborated)}</td>"
            "</tr>"
        )
    table_html = (
        '<table class="proof-table"><thead><tr>'
        "<th>Chunk</th><th>BM25 Rank</th><th>Dense Rank</th><th>Fused (RRF)</th><th>Reranked</th>"
        f"</tr></thead><tbody>{body_rows}</tbody></table>"
    )
    with st.expander("▾ RETRIEVAL PROOF — bm25 rank vs dense rank vs fused (rrf) rank", expanded=True):
        st.markdown(table_html, unsafe_allow_html=True)


def render_expanded_queries(variants: list[str]) -> None:
    """
    Takes: the expanded_queries list from a chat response — alternate
    phrasings app/graph/nodes.py's expand_node generated and actually
    retrieved on, alongside the rewritten query itself.
    Returns: nothing — renders a small expander listing them, or nothing at
    all if the list is empty (expansion is best-effort: an empty list is a
    real, valid outcome — either the LLM call failed and expand_node fell
    back silently, or there was nothing to show; either way there's no
    proof to display, so no expander claiming otherwise).
    Use this: from render_chat, right before render_retrieval_proof — shown
    first because expand runs before retrieve in the actual pipeline order.
    """
    if not variants:
        return
    with st.expander(f"🔀 {len(variants)} query variant(s) also searched", expanded=False):
        for variant in variants:
            st.caption(f"• {variant}")


def send_chat_message(query: str) -> None:
    """
    Takes: the user's question.
    Returns: nothing — appends the user turn and the API's reply (or an
    error/blocked message) to session_state, then reruns to redraw.
    Use this: from both the chat_input box and the suggested-question
    buttons, so both paths go through the same request/response handling.

    Renders a pulsing "pipeline running" placeholder before the blocking
    POST /api/v1/chat call, and lets it get overwritten by the final
    st.rerun() once a real answer (or error) exists — see the .stage.pending
    CSS rule's comment for why every stage pulses together instead of
    lighting up one at a time.
    """
    st.session_state.messages.append({"role": "user", "content": query})

    pending = st.empty()
    with pending.container():
        pending_stages = '<span class="stage-arrow">→</span>'.join(
            f'<span class="stage pending">{stage}</span>' for stage in PIPELINE_STAGES
        )
        st.markdown(f'<div class="pipeline">{pending_stages}</div>', unsafe_allow_html=True)
        st.caption("Running the pipeline — one blocking API call, so every stage lights up together, not step-by-step.")

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={"query": query, "conversation_id": st.session_state.conversation_id},
            timeout=120,
        )
    except requests.RequestException as e:
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Couldn't reach the API: {e}"})
        st.rerun()
        return

    if response.status_code == 200:
        data = response.json()
        st.session_state.conversation_id = data["conversation_id"]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "retrieval_count": data["retrieval_count"],
                "reranked_count": data["reranked_count"],
                "latency_ms": data["latency_ms"],
                "retrieval_proof": data.get("retrieval_proof", []),
                "expanded_queries": data.get("expanded_queries", []),
                "answer_model": data.get("answer_model", ""),
                "answer_provider": data.get("answer_provider", ""),
            }
        )
    else:
        # response.json() isn't guaranteed here even though the API's own
        # unhandled-exception handler (app/main.py) always returns JSON now
        # — defensive on principle, since this client shouldn't assume that
        # holds for every possible failure mode (a proxy in front of the
        # API, a connection that drops mid-response, etc.). This is exactly
        # what broke before that handler existed: a raw 500's plain-text
        # "Internal Server Error" body made response.json() itself raise,
        # replacing the real error with an unrelated JSONDecodeError.
        try:
            body = response.json()
        except requests.exceptions.JSONDecodeError:
            body = {}

        if response.status_code == 500:
            # Shape from app/main.py's global exception handler:
            # {"error": "internal_error", "detail": "<str(exc)>"}.
            reason = body.get("detail", response.text or "unknown error")
            message = f"⚠️ Something went wrong on the server ({reason})."
        else:
            # Shape from a deliberate HTTPException, e.g. guardrail blocks:
            # {"detail": {"error": "...", "reason": "..."}}.
            detail = body.get("detail", {})
            reason = detail.get("reason", "unknown") if isinstance(detail, dict) else detail
            message = f"🚫 This request was blocked ({reason})."

        st.session_state.messages.append({"role": "assistant", "content": message})
    st.rerun()


def render_chat() -> None:
    render_pipeline_explainer()

    if not st.session_state.messages:
        st.markdown("#### 👋 Ask anything about your knowledge base")
        starter_questions = [
            "What topics are covered in the knowledge base?",
            "Summarize the most recently uploaded document.",
            "What sources support your last answer?",
            "Explain the main idea of the uploaded material in simple terms.",
        ]
        cols = st.columns(2)
        for i, starter in enumerate(starter_questions):
            if cols[i % 2].button(starter, use_container_width=True, key=f"starter_{i}"):
                send_chat_message(starter)

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-row user"><div class="user-bubble">{html.escape(message["content"])}</div></div>',
                unsafe_allow_html=True,
            )
            continue

        # Assistant messages render as a plain left-aligned block (stage
        # pills + accent-bordered answer text), not a bubble — matches the
        # reference mockup, where only the user's turn gets a boxed shape.
        if "latency_ms" in message:
            # Shown after the fact, all stages "done" — /api/v1/chat is a
            # single blocking call, not a stream, so there's no honest way
            # to light these up one at a time as they actually happen
            # without restructuring the endpoint. This still reflects the
            # real stages app/graph/build_graph.py ran for this answer.
            stage_html = '<span class="stage-arrow">→</span>'.join(
                f'<span class="stage">{stage}</span>' for stage in PIPELINE_STAGES
            )
            st.markdown(f'<div class="pipeline">{stage_html}</div>', unsafe_allow_html=True)

        answer_html = html.escape(message["content"]).replace("\n", "<br>")
        st.markdown(f'<div class="answer-text">{answer_html}</div>', unsafe_allow_html=True)

        if "latency_ms" in message:
            # Which model actually answered, shown here rather than
            # asserted anywhere fixed — automatic Groq-model and
            # Groq->Ollama fallback (app/llm/llm_gateway.py) both mean the
            # answering model isn't predictable from settings alone, so
            # this is the one place a viewer can tell for sure.
            model_html = ""
            if message.get("answer_model"):
                model_html = f'<span>model <b>{html.escape(message["answer_provider"])}/{html.escape(message["answer_model"])}</b></span>'
            st.markdown(
                '<div class="meta-row">'
                f'<span>latency <b>{message["latency_ms"] / 1000:.1f}s</b></span>'
                f'<span>chunks retrieved <b>{message["retrieval_count"]}</b></span>'
                f'<span>reranked to <b>{message["reranked_count"]}</b></span>'
                "<span>guardrails <b>pass</b></span>"
                f"{model_html}"
                "</div>",
                unsafe_allow_html=True,
            )

        if message.get("expanded_queries"):
            render_expanded_queries(message["expanded_queries"])

        if message.get("retrieval_proof"):
            render_retrieval_proof(message["retrieval_proof"])

        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s)"):
                for source in message["sources"]:
                    st.caption(f"• {source}")

    if st.session_state.messages:
        # Same reset as the sidebar's "New chat" — placed below the
        # transcript, right-aligned, right above the input: it acts on
        # what's above it, so it reads naturally at the end of the chat
        # rather than floating above content it hasn't happened yet.
        _, clear_col = st.columns([6, 1])
        if clear_col.button("🗑️  Clear chat", key="clear_chat_main", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    if prompt := st.chat_input("Ask a question about your documents..."):
        send_chat_message(prompt)


# --- Upload ----------------------------------------------------------------


def render_upload() -> None:
    st.markdown("#### 📤 Upload a document")
    st.caption("PDF, TXT, DOCX, or an image (PNG/JPG/JPEG/TIFF/BMP — read via OCR).")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "tiff", "bmp"]
    )
    chunking_strategy = st.segmented_control(
        "Chunking strategy", options=["recursive", "semantic"], default="recursive"
    )
    st.caption(
        "Recursive splits by character count — fast, predictable. "
        "Semantic splits at topic boundaries — slower, better for documents that jump between subjects."
    )

    if st.button("Index this document", type="primary", disabled=uploaded_file is None):
        with st.spinner("Loading, chunking, and indexing…"):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/upload",
                    files=files,
                    params={"chunking_strategy": chunking_strategy or "recursive"},
                    timeout=180,
                )
            except requests.RequestException as e:
                st.error(f"Couldn't reach the API: {e}")
                return

        if response.status_code == 200:
            data = response.json()
            st.session_state.uploads.insert(0, data)
            st.session_state.last_upload_message = (
                f"Indexed **{data['chunks_indexed']}** chunk(s) from **{data['filename']}** "
                f"using the {data['chunking_strategy']} chunker."
            )
            # Rerun rather than just calling st.success() here — the sidebar's
            # Documents/Chunks counts are computed once at the top of the
            # script, so without a rerun they'd keep showing the pre-upload
            # numbers until some unrelated interaction happened to refresh them.
            st.rerun()
        else:
            st.error(f"Upload failed: {response.json().get('detail', response.text)}")

    if "last_upload_message" in st.session_state:
        st.success(st.session_state.pop("last_upload_message"))

    render_indexed_documents()


def render_indexed_documents() -> None:
    """
    Lists every active source currently in the corpus (pre-loaded samples
    and past uploads alike — there's no distinction once indexed) with two
    actions per row: 🙈 hide (temporary — DELETE .../documents?permanent=false,
    embeddings kept) and 🗑️ delete (permanent — the original behavior, embeddings
    gone). A separate expander below lists hidden documents with a ↩️ restore
    button. Lives on the Upload page rather than the sidebar's "Pre-loaded
    corpus" list, which stays a plain read-only list matching the reference
    mockup — management actions belong on the page whose whole purpose is
    managing the corpus, not bolted onto navigation chrome.
    """
    st.divider()
    st.markdown("###### 📚 Indexed documents")
    st.caption("Everything currently searchable. Hide keeps the embeddings (instant restore); delete removes them for good.")

    try:
        stats_response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5).json()
        sources = stats_response.get("sources", [])
        hidden_sources = stats_response.get("hidden_sources", [])
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API: {e}")
        return

    if not sources:
        st.caption("Nothing indexed yet.")
    for source in sources:
        name_col, hide_col, delete_col = st.columns([5, 1, 1])
        name_col.markdown(f"`{source}`")
        if hide_col.button("🙈", key=f"hide_{source}", help=f"Temporarily hide {source} (embeddings kept — restore anytime)"):
            _delete_document(source, permanent=False)
        if delete_col.button("🗑️", key=f"delete_{source}", help=f"Permanently delete {source} (embeddings gone)"):
            _delete_document(source, permanent=True)

    # Hidden documents get their own section rather than mixing a "restore"
    # button into the same rows above — they're not currently searchable,
    # so listing them among "Indexed documents" (this function's own
    # premise, per its docstring) would be misleading.
    if hidden_sources:
        with st.expander(f"🙈 Hidden documents ({len(hidden_sources)})", expanded=False):
            for source in hidden_sources:
                name_col, restore_col = st.columns([6, 1])
                name_col.markdown(f"`{source}`")
                if restore_col.button("↩️", key=f"restore_{source}", help=f"Restore {source}"):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/documents/restore", params={"source": source}, timeout=30
                        )
                    except requests.RequestException as e:
                        st.error(f"Couldn't reach the API: {e}")
                    else:
                        if response.status_code == 200:
                            chunks_restored = response.json()["chunks_restored"]
                            st.session_state.last_upload_message = f"Restored **{chunks_restored}** chunk(s) of **{source}**."
                            st.rerun()
                        else:
                            st.error(f"Restore failed: {response.json().get('detail', response.text)}")


def _delete_document(source: str, permanent: bool) -> None:
    """
    Takes: the source filename and whether to hard- or soft-delete it.
    Returns: nothing — calls DELETE /api/v1/documents and updates session
    state/reruns on success, same as the inline restore handling above.
    Use this: from render_indexed_documents()'s hide/delete buttons — pulled
    into its own function since both buttons need this exact request/error/
    rerun sequence, differing only in the `permanent` query param.
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/documents", params={"source": source, "permanent": permanent}, timeout=30
        )
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API: {e}")
        return

    if response.status_code == 200:
        chunks_deleted = response.json()["chunks_deleted"]
        verb = "Deleted" if permanent else "Hid"
        if permanent:
            st.session_state.uploads = [u for u in st.session_state.uploads if u["filename"] != source]
        st.session_state.last_upload_message = f"{verb} **{chunks_deleted}** chunk(s) of **{source}**."
        st.rerun()
    else:
        st.error(f"{'Delete' if permanent else 'Hide'} failed: {response.json().get('detail', response.text)}")


# --- Search ------------------------------------------------------------------


def render_search() -> None:
    st.markdown("#### 🔍 Search the knowledge base")
    st.caption("Retrieval only — hybrid (BM25 + dense) fusion and reranking, no LLM call.")

    query = st.text_input("Query", placeholder="e.g. What is reciprocal rank fusion?")
    k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching…"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/search", json={"query": query, "k": k}, timeout=60
                )
            except requests.RequestException as e:
                st.error(f"Couldn't reach the API: {e}")
                return

        if response.status_code == 200:
            results = response.json()["results"]
            if not results:
                st.info("No results — try uploading a document first.")
            for result in results:
                with st.container(border=True):
                    result_col, score_col = st.columns([4, 1])
                    result_col.markdown(f"**{result['source']}**")
                    score_col.badge(f"score {result['score']:.3f}", color="blue")
                    st.write(result["text"])
        else:
            st.error(f"Search failed: {response.json().get('detail', response.text)}")


# --- Page dispatch -----------------------------------------------------------

if st.session_state.page == "Upload":
    render_upload()
elif st.session_state.page == "Search & Retrieval":
    render_search()
else:
    render_chat()
