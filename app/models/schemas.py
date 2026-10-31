from typing import List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

# --- Query Schemas ---
class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=2, 
        max_length=1000, 
        description="The natural language question to ask the documentation assistant.",
        examples=["How do I declare optional query parameters with default values in FastAPI?"]
    )
    thread_id: Optional[str] = Field(
        default=None, 
        description="Optional session / thread ID for multi-turn conversation memory.",
        examples=["session-user-123"]
    )

class CitationSource(BaseModel):
    source: str = Field(..., description="Document filename or URL", examples=["path_and_query_parameters.md"])
    section: Optional[str] = Field(default=None, description="Section header or title", examples=["Query Parameters"])
    snippet: str = Field(..., description="Excerpt from the source document")
    relevance_score: Optional[float] = Field(default=None, description="Similarity or grader score")

class QueryResponse(BaseModel):
    question: str = Field(..., description="Original question submitted")
    query_type: str = Field(..., description="Classified query type (conceptual, how-to, troubleshooting, api-reference)")
    answer: str = Field(..., description="Generated answer grounded in documentation with citations")
    sources: List[CitationSource] = Field(default_factory=list, description="Citations and references to source documents")
    retries_used: int = Field(default=0, description="Number of query expansion/rewriting retries performed")
    is_fallback: bool = Field(default=False, description="True if answer required web search or fallback response")
    is_grounded: bool = Field(default=True, description="True if passed hallucination/groundedness verification")
    thread_id: Optional[str] = Field(default=None, description="Session thread ID used for conversation history")

# --- Ingestion Schemas ---
class IngestURLRequest(BaseModel):
    urls: List[str] = Field(
        ..., 
        min_length=1, 
        description="List of documentation URLs to scrape, chunk, and index.",
        examples=[["https://fastapi.tiangolo.com/tutorial/first-steps/"]]
    )

class IngestResponse(BaseModel):
    status: str = Field(..., examples=["success"])
    message: str = Field(..., examples=["Ingested 4 documents, generated 18 chunks."])
    indexed_documents: List[str] = Field(default_factory=list)
    total_chunks_added: int = Field(default=0)

# --- Document Listing Schemas ---
class DocumentChunkInfo(BaseModel):
    id: str
    source: str
    section: Optional[str] = None
    character_count: int

class DocumentInfo(BaseModel):
    source: str
    chunk_count: int
    sections: List[str]

class DocumentsListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]

# --- Feedback Schemas ---
class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Identifier of the query being evaluated", examples=["q-8f4b-2026"])
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or thumbs down rating", examples=["up"])
    comment: Optional[str] = Field(default=None, description="Optional user comment", examples=["Very accurate code snippet!"])
    question: Optional[str] = Field(default=None, description="The original question evaluated")
    answer: Optional[str] = Field(default=None, description="The answer evaluated")

class FeedbackResponse(BaseModel):
    status: str = Field(default="recorded")
    feedback_id: str
    message: str = Field(default="Thank you for your feedback!")

# --- Health Check Schema ---
class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    indexed_chunks: int
    version: str
