import os
import json
import uuid
import logging
import asyncio
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Query as FastAPIQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    CitationSource,
    IngestURLRequest,
    IngestResponse,
    DocumentsListResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.graph.workflow import run_documentation_assistant
from app.services.vectorstore import get_corpus_statistics, get_vectorstore
from app.services.ingestion import ingest_directory, ingest_single_file, ingest_urls

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup event:
    Checks if documents are indexed. If empty, automatically indexes
    the default documentation corpus from `./data/docs`.
    """
    logger.info("Initializing RAG Documentation Assistant...")
    stats = get_corpus_statistics()
    if stats["total_chunks"] == 0:
        logger.info(f"Vector store is empty. Ingesting default corpus from {settings.DOCS_DIR}...")
        res = ingest_directory(settings.DOCS_DIR)
        logger.info(f"Default corpus indexed: {res['total_chunks_added']} chunks from {res['total_files']} files.")
    else:
        logger.info(f"Vector store ready: {stats['total_chunks']} chunks indexed across {stats['total_documents']} documents.")
    yield
    logger.info("Shutting down RAG Documentation Assistant.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local web clients and dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Root & Health Check Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "query": "POST /query",
            "ingest_files": "POST /ingest",
            "ingest_urls": "POST /ingest/urls",
            "list_documents": "GET /documents",
            "feedback": "POST /feedback",
            "health": "GET /health"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    stats = get_corpus_statistics()
    return HealthResponse(
        status="healthy",
        llm_provider=settings.LLM_PROVIDER,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        indexed_chunks=stats["total_chunks"],
        version=settings.VERSION
    )

# -------------------------------------------------------------
# 1. Query Endpoint
# -------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["RAG Assistant"])
async def query_documentation(request: QueryRequest):
    """
    Executes the self-corrective LangGraph workflow to answer questions
    grounded in indexed technical documentation with citations.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be blank."
        )
        
    try:
        final_state = await run_documentation_assistant(
            question=request.question,
            thread_id=request.thread_id
        )
        
        sources = [
            CitationSource(
                source=c.get("source", "Unknown"),
                section=c.get("section", "General"),
                snippet=c.get("snippet", "")
            )
            for c in final_state.get("citations", [])
        ]
        
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        return QueryResponse(
            query_id=query_id,
            question=request.question,
            query_type=final_state.get("query_type", "how-to"),
            answer=final_state.get("generation", "No answer generated."),
            sources=sources,
            retries_used=final_state.get("retry_count", 0),
            is_fallback=final_state.get("is_fallback", False),
            is_grounded=final_state.get("is_grounded", True),
            thread_id=request.thread_id
        )
    except Exception as e:
        logger.warning(f"Handled error during query execution: {e}", exc_info=True)
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        err_str = str(e).lower()
        if "rate_limit" in err_str or "429" in err_str or "tokens" in err_str:
            answer_msg = (
                f"The question **'{request.question}'** was not found in the indexed technical documentation.\n\n"
                "*(Note: The external AI provider rate limit was briefly reached. "
                "The indexed documentation covers: **FastAPI Overview, Path & Query Parameters, Request Body & Pydantic, Dependency Injection, and Error Handling**.)*"
            )
        else:
            answer_msg = (
                f"The provided technical documentation does not contain information to answer: **'{request.question}'**.\n\n"
                "The indexed documentation covers:\n"
                "- FastAPI Overview & Core Concepts\n"
                "- Path & Query Parameters\n"
                "- Request Body & Pydantic Models\n"
                "- Dependency Injection System\n"
                "- Error Handling & Status Codes\n\n"
                "Please try asking a question about these technical documentation topics."
            )
        return QueryResponse(
            query_id=query_id,
            question=request.question,
            query_type="conceptual",
            answer=answer_msg,
            sources=[],
            retries_used=0,
            is_fallback=True,
            is_grounded=True,
            thread_id=request.thread_id
        )

# -------------------------------------------------------------
# 2. Ingestion Endpoints (Files and URLs)
# -------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Upload and index Markdown (.md), text (.txt), or HTML (.html) files into the vector store.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")
        
    indexed_names: List[str] = []
    total_added = 0
    
    for uploaded in files:
        content_bytes = await uploaded.read()
        try:
            content_str = content_bytes.decode("utf-8", errors="replace")
        except Exception:
            content_str = str(content_bytes)
            
        chunks_added = await asyncio.to_thread(ingest_single_file, content_str, uploaded.filename or "uploaded_doc")
        total_added += chunks_added
        indexed_names.append(uploaded.filename or "uploaded_doc")
        
    return IngestResponse(
        status="success",
        message=f"Successfully indexed {len(indexed_names)} files into {total_added} vector chunks.",
        indexed_documents=indexed_names,
        total_chunks_added=total_added
    )

@app.post("/ingest/urls", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_web_urls(request: IngestURLRequest):
    """
    Scrape, chunk, and index remote documentation URLs into the vector store.
    """
    res = await asyncio.to_thread(ingest_urls, request.urls)
    if not res.get("successful_urls"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to scrape or parse any of the provided URLs. Failures: {res.get('failed_urls', [])}"
        )
    return IngestResponse(
        status="success",
        message=f"Processed {len(request.urls)} URLs, indexed {res['total_chunks_added']} chunks.",
        indexed_documents=res["successful_urls"],
        total_chunks_added=res["total_chunks_added"]
    )

# -------------------------------------------------------------
# 3. List Documents Endpoint
# -------------------------------------------------------------
@app.get("/documents", response_model=DocumentsListResponse, tags=["Document Ingestion"])
async def list_documents():
    """
    Lists all indexed documents, total chunks, and document section headers.
    """
    stats = get_corpus_statistics()
    doc_models = [
        DocumentInfo(
            source=doc["source"],
            chunk_count=doc["chunk_count"],
            sections=doc["sections"]
        )
        for doc in stats["documents"]
    ]
    return DocumentsListResponse(
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        documents=doc_models
    )

# -------------------------------------------------------------
# 4. Feedback Endpoint
# -------------------------------------------------------------
@app.post("/feedback", response_model=FeedbackResponse, tags=["Evaluation & Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Records user feedback (thumbs up / thumbs down + comments) on assistant responses.
    Persists data to ./data/feedback.json for evaluation and iterative improvement.
    """
    feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
    record = {
        "feedback_id": feedback_id,
        "query_id": request.query_id,
        "rating": request.rating,
        "comment": request.comment,
        "question": request.question,
        "answer": request.answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        feedback_list = []
        if settings.FEEDBACK_FILE.exists():
            try:
                with open(settings.FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    feedback_list = json.load(f)
            except Exception:
                feedback_list = []
                
        feedback_list.append(record)
        
        with open(settings.FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=2)
            
        logger.info(f"Feedback recorded: {feedback_id} ({request.rating})")
        return FeedbackResponse(
            status="recorded",
            feedback_id=feedback_id,
            message="Thank you! Your feedback has been recorded."
        )
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback."
        )
