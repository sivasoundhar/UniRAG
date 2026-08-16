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
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="UniRAG", page_icon="📚", layout="wide")

# Custom CSS for two things only: the header banner, and the pipeline-trace
# / meta-row readouts on chat answers (monospace "stage" pills, like a build
# log) — everything else on this page is native Streamlit components
# (st.container(border=True), st.chat_message, st.segmented_control,
# st.badge), not CSS-styled divs. Colors are chosen to hold up in both
# light and dark theme rather than assuming one.
st.markdown(
    """
    <style>
    .unirag-header {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f766e 0%, #134e4a 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .unirag-header h1 { margin: 0; font-size: 1.6rem; }
    .unirag-header p { margin: 0.25rem 0 0 0; opacity: 0.9; }

    .pipeline { display:flex; align-items:center; gap:5px; margin: 2px 0 10px 0; flex-wrap:wrap; }
    .stage {
        font-family: 'Courier New', monospace; font-size: 11px; padding: 3px 9px;
        border-radius: 5px; border: 1px solid rgba(13,148,136,0.35);
        color: #0d9488; background: rgba(13,148,136,0.10);
    }
    .stage-arrow { color: rgba(128,128,128,0.5); font-size: 11px; }

    .meta-row {
        display:flex; gap:16px; margin: 4px 0 8px 0; flex-wrap: wrap;
        font-family: 'Courier New', monospace; font-size: 11.5px; color: rgba(128,128,128,0.95);
    }
    .meta-row b { color: #b45309; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

PIPELINE_STAGES = ["rewrite", "retrieve", "fuse", "rerank", "compress", "generate"]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "uploads" not in st.session_state:
    st.session_state.uploads = []


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 UniRAG")
    st.caption("Reusable AI Knowledge Core")

    page = st.segmented_control(
        "Navigate",
        options=["💬 Chat", "📤 Upload", "🔍 Search"],
        default="💬 Chat",
        label_visibility="collapsed",
    )

    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"conversation: `{st.session_state.conversation_id[:8]}…`")

    st.divider()
    st.markdown("**Domain**")
    st.caption("Preview only — doesn't filter retrieval yet.")
    domain = st.radio(
        "Domain",
        options=["General", "Engineering Copilot", "Medical Copilot"],
        label_visibility="collapsed",
    )
    if domain != "General":
        st.caption(
            f"↳ UniRAG is the shared retrieval core; a real "
            f"'{domain}' would be its own app built on this API, not a mode switch."
        )

    st.divider()
    st.markdown("**System status**")

    api_online = False
    try:
        requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5).raise_for_status()
        api_online = True
        st.badge("API online", color="green")
    except requests.RequestException:
        st.badge("API offline", color="red")
        st.caption(f"Can't reach {API_BASE_URL}")

    if api_online:
        try:
            corpus_stats = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5).json()
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Documents", corpus_stats["document_count"])
            stat_col2.metric("Chunks", corpus_stats["chunk_count"])
        except requests.RequestException:
            pass

    st.divider()
    st.caption(
        "retrieval `hybrid (BM25+dense)`  \n"
        "rerank `cross-encoder`  \n"
        "llm `groq / ollama`  \n"
        "guardrails `nemo, regex-based`"
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


def send_chat_message(query: str) -> None:
    """
    Takes: the user's question.
    Returns: nothing — appends the user turn and the API's reply (or an
    error/blocked message) to session_state, then reruns to redraw.
    Use this: from both the chat_input box and the suggested-question
    buttons, so both paths go through the same request/response handling.
    """
    st.session_state.messages.append({"role": "user", "content": query})
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
            }
        )
    else:
        detail = response.json().get("detail", {})
        reason = detail.get("reason", "unknown") if isinstance(detail, dict) else detail
        st.session_state.messages.append(
            {"role": "assistant", "content": f"🚫 This request was blocked ({reason})."}
        )
    st.rerun()


def render_chat() -> None:
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
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "latency_ms" in message:
                # Shown after the fact, all stages "done" — /api/v1/chat is a
                # single blocking call, not a stream, so there's no honest way
                # to light these up one at a time as they actually happen
                # without restructuring the endpoint. This still reflects the
                # real stages app/graph/build_graph.py ran for this answer.
                stage_html = '<span class="stage-arrow">→</span>'.join(
                    f'<span class="stage">{stage}</span>' for stage in PIPELINE_STAGES
                )
                st.markdown(f'<div class="pipeline">{stage_html}</div>', unsafe_allow_html=True)

            st.markdown(message["content"])

            if message["role"] == "assistant" and "latency_ms" in message:
                st.markdown(
                    '<div class="meta-row">'
                    f'<span>latency <b>{message["latency_ms"] / 1000:.1f}s</b></span>'
                    f'<span>chunks retrieved <b>{message["retrieval_count"]}</b></span>'
                    f'<span>reranked to <b>{message["reranked_count"]}</b></span>'
                    '<span>guardrails <b>pass</b></span>'
                    "</div>",
                    unsafe_allow_html=True,
                )

            if message.get("sources"):
                with st.expander(f"📎 {len(message['sources'])} source(s)"):
                    for source in message["sources"]:
                        st.caption(f"• {source}")

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

    if st.session_state.uploads:
        st.divider()
        st.markdown("###### This session's uploads")
        for upload_record in st.session_state.uploads:
            with st.container(border=True):
                st.markdown(f"**{upload_record['filename']}**")
                st.caption(f"{upload_record['chunks_indexed']} chunk(s) · {upload_record['chunking_strategy']} chunking")


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

if page == "📤 Upload":
    render_upload()
elif page == "🔍 Search":
    render_search()
else:
    render_chat()
