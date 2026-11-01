"""
Streamlit Web Frontend for RAG Technical Documentation Assistant.
Provides interactive Q&A, source citation inspection, dynamic document ingestion,
and one-click answer feedback logging with a Technical Green theme.
"""

import os
import sys
import time
import subprocess
import streamlit as st
import httpx
import uuid

# Propagate Streamlit Cloud secrets to os.environ so backend and subprocesses inherit them
try:
    for k, v in st.secrets.items():
        if isinstance(v, str) and k not in os.environ:
            os.environ[k] = v
except Exception:
    pass

def get_backend_url() -> str:
    # 1. Check Streamlit secrets or OS environment
    url = None
    try:
        url = st.secrets.get("API_BASE_URL", None)
    except Exception:
        pass
    if not url:
        url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    return url.rstrip("/")

API_BASE_URL = get_backend_url()

@st.cache_resource
def ensure_backend_alive(url: str):
    """If pointing to localhost and not reachable, automatically launch FastAPI in the background."""
    if "127.0.0.1" in url or "localhost" in url:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        # Attempt to launch uvicorn in the background
        kwargs = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs
            )
            for _ in range(8):
                time.sleep(1)
                try:
                    r = httpx.get(f"{url}/health", timeout=1.0)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
    return False

ensure_backend_alive(API_BASE_URL)


st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📗",
    layout="wide"
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:6]}"

# Inject Technical Green Theme CSS
st.markdown("""
<style>
/* Technical Green Palette */
:root {
    --tech-green: #10B981;
    --tech-green-dark: #059669;
    --tech-green-light: #34D399;
    --tech-green-bg: rgba(16, 185, 129, 0.08);
    --tech-border: rgba(16, 185, 129, 0.35);
}

/* Header Banner */
.tech-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.05) 100%);
    border: 1px solid var(--tech-border);
    border-left: 6px solid var(--tech-green);
    padding: 16px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);
}

.tech-banner h1 {
    color: #10B981 !important;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.5px;
}

.tech-banner p {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 6px 0 0 0;
    font-weight: 500;
}

/* Technical Green Badges */
.tech-pill {
    display: inline-block;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 20px;
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
    margin-right: 6px;
}

.tech-pill-gray {
    display: inline-block;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 20px;
    background: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-right: 6px;
}

/* Glowing green status dot */
.pulse-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #10B981;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 10px #10B981;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--tech-border);
}

/* Buttons */
.stButton > button {
    border: 1px solid var(--tech-border) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover {
    border-color: var(--tech-green) !important;
    color: var(--tech-green) !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.25) !important;
}

/* Tab Highlights */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--tech-green) !important;
    border-bottom-color: var(--tech-green) !important;
}

/* Expander custom border */
.streamlit-expanderHeader {
    border-left: 3px solid var(--tech-green) !important;
}
</style>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
<div class="tech-banner">
    <h1>📗 ⚡ Technical Documentation Assistant</h1>
    <p>Self-Corrective LangGraph Architecture with CRAG, Citation Grounding & FastAPI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - System Info and Settings
with st.sidebar:
    st.markdown("### ⚙️ Engine Console")
    try:
        health_resp = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.markdown(
                '<p style="color:#10B981; font-weight:600;"><span class="pulse-dot"></span>Backend API: Online</p>',
                unsafe_allow_html=True
            )
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Indexed Chunks", health_data.get("indexed_chunks", 0))
            with col_m2:
                st.metric("Version", health_data.get("version", "1.0"))
            st.info(f"LLM Provider: **{health_data.get('llm_provider', 'N/A').upper()}**")
        else:
            st.warning("Backend API responded with non-200 code.")
    except Exception:
        st.error("Backend API is unreachable at http://127.0.0.1:8000. Start the FastAPI server!")
        
    st.divider()
    with st.expander("🔑 API Credentials", expanded=not bool(os.getenv("GROQ_API_KEY"))):
        groq_key_input = st.text_input(
            "Groq API Key", 
            value=os.getenv("GROQ_API_KEY", ""), 
            type="password",
            help="Your Groq API Key (starts with gsk_)"
        )
        tavily_key_input = st.text_input(
            "Tavily API Key (Optional)", 
            value=os.getenv("TAVILY_API_KEY", ""), 
            type="password",
            help="Your Tavily API Key (starts with tvly-)"
        )
        if groq_key_input and groq_key_input != os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = groq_key_input
            st.success("Groq API Key saved!")
            st.rerun()
        if tavily_key_input and tavily_key_input != os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = tavily_key_input
            st.success("Tavily API Key saved!")
            st.rerun()

    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    if st.button("🗑️ Clear Session Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:6]}"
        st.rerun()

# Tabs for Main Interface
tab_chat, tab_docs, tab_workflow = st.tabs(["💬 Interactive Q&A", "📄 Document Corpus", "🔍 LangGraph Architecture"])

# -------------------------------------------------------------
# TAB 1: Interactive Q&A Chat
# -------------------------------------------------------------
with tab_chat:
    # Quick Starter Questions (shown when conversation is empty)
    if not st.session_state.messages:
        st.markdown("##### 💡 Example Questions (Click to Ask)")
        q_cols = st.columns(2)
        starters = [
            "⚡ How do path parameters work in FastAPI?",
            "🛡️ How do I validate request bodies with Pydantic?",
            "🔌 How does Dependency Injection work in FastAPI?",
            "⚠️ How do I handle HTTP errors and status codes?"
        ]
        for idx, starter in enumerate(starters):
            col = q_cols[idx % 2]
            with col:
                if st.button(starter, key=f"starter_{idx}", use_container_width=True):
                    # Clean out icon prefix for cleaner question
                    clean_q = starter.split(" ", 1)[1] if " " in starter else starter
                    st.session_state.starter_selected = clean_q
                    st.rerun()

    # Display Conversation Messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📗"):
            st.markdown(msg["content"])
            
            # Show metadata and citations for assistant messages
            if msg["role"] == "assistant":
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.markdown(f'<span class="tech-pill">🏷️ {msg.get("query_type", "how-to").upper()}</span>', unsafe_allow_html=True)
                with col2:
                    retries = msg.get("retries_used", 0)
                    st.markdown(f'<span class="tech-pill-gray">🔄 Retries: {retries}</span>', unsafe_allow_html=True)
                with col3:
                    if msg.get("is_grounded", True):
                        st.markdown('<span class="tech-pill">🛡️ ✅ Grounded</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="tech-pill-gray">⚠️ Unverified</span>', unsafe_allow_html=True)
                    
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📖 View {len(sources)} Verified Citations"):
                        for s_idx, s in enumerate(sources, 1):
                            st.markdown(f"**[{s_idx}] {s.get('source')}** *(Section: {s.get('section', 'General')})*")
                            st.caption(f"_{s.get('snippet', '')}_")
                            st.divider()
                            
                # Feedback buttons
                f_col1, f_col2, _ = st.columns([1, 1, 6])
                query_id = msg.get("query_id", f"q_{idx}")
                with f_col1:
                    if st.button("👍", key=f"up_{idx}"):
                        try:
                            httpx.post(
                                f"{API_BASE_URL}/feedback",
                                json={"query_id": query_id, "rating": "up", "question": msg.get("question")},
                                timeout=2.0
                            )
                            st.toast("Feedback recorded: Thumbs Up!")
                        except Exception as e:
                            st.toast(f"Feedback failed: {e}")
                with f_col2:
                    if st.button("👎", key=f"down_{idx}"):
                        try:
                            httpx.post(
                                f"{API_BASE_URL}/feedback",
                                json={"query_id": query_id, "rating": "down", "question": msg.get("question")},
                                timeout=2.0
                            )
                            st.toast("Feedback recorded: Thumbs Down!")
                        except Exception as e:
                            st.toast(f"Feedback failed: {e}")

    # Handle starter button click if present
    prefilled_query = None
    if "starter_selected" in st.session_state:
        prefilled_query = st.session_state.pop("starter_selected")

    # Chat Input Box
    user_query = st.chat_input("Ask a technical documentation question (e.g., 'How do path parameters work in FastAPI?')...")
    active_query = prefilled_query or user_query

    if active_query:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_query)

        # Call FastAPI backend
        with st.chat_message("assistant", avatar="📗"):
            with st.spinner("Analyzing query, retrieving documents, and self-correcting..."):
                try:
                    payload = {
                        "question": active_query,
                        "thread_id": st.session_state.session_id,
                        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
                        "tavily_api_key": os.getenv("TAVILY_API_KEY", "")
                    }
                    response = httpx.post(f"{API_BASE_URL}/query", json=payload, timeout=60.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        query_type = data.get("query_type", "how-to")
                        sources = data.get("sources", [])
                        retries = data.get("retries_used", 0)
                        is_grounded = data.get("is_grounded", True)
                        query_id = data.get("query_id") or f"q_{uuid.uuid4().hex[:6]}"
                        
                        st.markdown(answer)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "question": active_query,
                            "query_id": query_id,
                            "query_type": query_type,
                            "sources": sources,
                            "retries_used": retries,
                            "is_grounded": is_grounded
                        })
                        st.rerun()
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API server: {e}")

# -------------------------------------------------------------
# TAB 2: Document Corpus Management
# -------------------------------------------------------------
with tab_docs:
    st.subheader("📑 Indexed Documentation Corpus")
    try:
        docs_resp = httpx.get(f"{API_BASE_URL}/documents", timeout=5.0)
        if docs_resp.status_code == 200:
            doc_data = docs_resp.json()
            st.markdown(
                f'<span class="tech-pill">📄 Total Documents: {doc_data["total_documents"]}</span>'
                f'<span class="tech-pill">🧩 Total Chunks: {doc_data["total_chunks"]}</span>',
                unsafe_allow_html=True
            )
            st.write("")
            
            for d in doc_data.get("documents", []):
                with st.expander(f"📄 {d['source']} ({d['chunk_count']} chunks)"):
                    st.write("**Document Sections:**")
                    for sec in d.get("sections", []):
                        st.write(f"- {sec}")
        else:
            st.warning("Failed to fetch documents from API.")
    except Exception as e:
        st.error(f"API Error: {e}")
        
    st.divider()
    st.subheader("➕ Ingest New Technical Documentation")
    
    col_upload, col_url = st.columns(2)
    with col_upload:
        st.write("##### Upload Local Files (.md, .txt)")
        uploaded_files = st.file_uploader("Upload Documentation Files", accept_multiple_files=True, type=["md", "txt", "html"])
        if st.button("Index Uploaded Files"):
            if uploaded_files:
                files_payload = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
                with st.spinner("Chunking and indexing documents..."):
                    resp = httpx.post(f"{API_BASE_URL}/ingest", files=files_payload, timeout=60.0)
                    if resp.status_code == 200:
                        st.success(resp.json().get("message", "Files indexed successfully!"))
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {resp.text}")
            else:
                st.warning("Please select at least one file to upload.")
                
    with col_url:
        st.write("##### Scrape Documentation URL")
        doc_url = st.text_input("Enter Documentation URL (e.g., https://fastapi.tiangolo.com/tutorial/)")
        if st.button("Index Web URL"):
            if doc_url.strip():
                with st.spinner("Scraping and indexing URL..."):
                    resp = httpx.post(f"{API_BASE_URL}/ingest/urls", json={"urls": [doc_url.strip()]}, timeout=60.0)
                    if resp.status_code == 200:
                        st.success(resp.json().get("message", "URL indexed successfully!"))
                        st.rerun()
                    else:
                        st.error(f"URL indexing failed: {resp.text}")
            else:
                st.warning("Please enter a valid URL.")

# -------------------------------------------------------------
# TAB 3: LangGraph Architecture Walkthrough
# -------------------------------------------------------------
with tab_workflow:
    st.subheader("🧠 Self-Corrective LangGraph Architecture")
    st.markdown("""
    This project implements an **Agentic Self-Corrective RAG (CRAG / Self-RAG)** pipeline:
    
    1. **Query Analysis:** Classifies user intent (`conceptual`, `how-to`, `troubleshooting`, `api-reference`) and rewrites raw query with technical synonyms.
    2. **Retrieval:** Performs vector similarity search over ChromaDB embeddings with Markdown-preserved metadata.
    3. **Document Grading:** An LLM evaluates each chunk individually for relevance, discarding noise.
    4. **Conditional Routing:**
       - *Relevant chunks found?* $\\rightarrow$ Proceed to generation.
       - *Zero relevant chunks and retries remain?* $\\rightarrow$ Reformulate query and re-retrieve.
       - *Retries exhausted?* $\\rightarrow$ Fall back to Web Search (Tavily/DuckDuckGo) or explicit context refusal.
    5. **Answer Generation:** Strictly grounded answer with inline file and section citations.
    6. **Hallucination Verification (Self-RAG):** Validates that all factual claims in the answer are supported by retrieved context.
    """)
    
    st.code("""
    [User Question]
           │
           ▼
    [Query Analysis] (Classify Intent + Expand Keywords)
           │
           ▼
    [Vector Retrieval] (ChromaDB top-k)
           │
           ▼
    [Document Grading] (Filter Irrelevant Chunks)
           │
           ├─► (Relevant chunks >= 1) ───────► [Generation & Citations] ──► [Hallucination Check] ──► END
           │
           ├─► (0 relevant & retries < 2) ───► [Query Rewriter] ──► [Vector Retrieval] (Loop)
           │
           └─► (0 relevant & retries >= 2) ──► [Web Search Fallback] ──► [Generation] ──► END
    """, language="text")
