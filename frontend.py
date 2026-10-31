"""
Streamlit Web Frontend for RAG Technical Documentation Assistant.
Provides interactive Q&A, source citation inspection, dynamic document ingestion,
and one-click answer feedback logging.
"""

import streamlit as st
import httpx
import uuid

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:6]}"

st.title("📚 Technical Documentation Assistant")
st.caption("Powered by LangGraph Self-Corrective RAG & FastAPI")

# Sidebar - System Info and Settings
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_resp = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("Backend API: Online")
            st.metric("Indexed Chunks", health_data.get("indexed_chunks", 0))
            st.info(f"LLM Provider: **{health_data.get('llm_provider', 'N/A').upper()}**")
        else:
            st.warning("Backend API responded with non-200 code.")
    except Exception:
        st.error("Backend API is unreachable at http://127.0.0.1:8000. Start the FastAPI server!")
        
    st.divider()
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:6]}"
        st.rerun()

# Tabs for Main Interface
tab_chat, tab_docs, tab_workflow = st.tabs(["💬 Interactive Q&A", "📄 Document Corpus", "🔍 LangGraph Architecture"])

# -------------------------------------------------------------
# TAB 1: Interactive Q&A Chat
# -------------------------------------------------------------
with tab_chat:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show metadata and citations for assistant messages
            if msg["role"] == "assistant":
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.caption(f"🏷️ Intent: **{msg.get('query_type', 'N/A')}**")
                with col2:
                    st.caption(f"🔄 Retries: **{msg.get('retries_used', 0)}**")
                with col3:
                    grounded_icon = "✅ Grounded" if msg.get("is_grounded", True) else "⚠️ Verification Warning"
                    st.caption(f"🛡️ {grounded_icon}")
                    
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

    # Chat Input Box
    user_query = st.chat_input("Ask a technical documentation question (e.g., 'How do path parameters work in FastAPI?')...")
    if user_query:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Call FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, retrieving documents, and self-correcting..."):
                try:
                    payload = {
                        "question": user_query,
                        "thread_id": st.session_state.session_id
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
                            "question": user_query,
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
            st.write(f"Total Indexed Documents: **{doc_data['total_documents']}** | Total Chunks: **{doc_data['total_chunks']}**")
            
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
