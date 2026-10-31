from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Represents the internal state of our LangGraph self-corrective RAG workflow.
    Tracks query progression, retrieved & filtered documents, retry counts,
    citations, and hallucination flags.
    """
    question: str                      # Original raw user input
    query_type: str                    # conceptual | how-to | troubleshooting | api-reference
    transformed_query: str             # Query expanded/optimized for vector search
    documents: List[Document]          # Raw chunks retrieved from vector store
    filtered_docs: List[Document]      # Chunks confirmed relevant by Document Grader
    generation: str                    # LLM-generated answer grounded in context
    citations: List[Dict[str, Any]]    # Extracted source citations (file, section, snippet)
    retry_count: int                   # Current retry attempt counter
    max_retries: int                   # Maximum allowed rewrite loops
    is_fallback: bool                  # Flag indicating web search or insufficient fallback
    is_grounded: bool                  # Hallucination check result (True if grounded)
    chat_history: List[Dict[str, str]] # Multi-turn conversational history
