import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

class Settings(BaseModel):
    PROJECT_NAME: str = "RAG Technical Documentation Assistant"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Self-corrective LangGraph RAG assistant for technical documentation "
        "with adaptive query rewriting, document grading, citation grounding, and FastAPI serving."
    )
    
    # Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCS_DIR: Path = PROJECT_ROOT / "data" / "docs"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"
    FEEDBACK_FILE: Path = PROJECT_ROOT / "data" / "feedback.json"
    
    # LLM Settings
    # Supported: "groq", "gemini", "openai", "mock"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Embeddings
    # "huggingface" (offline, fast, free) or "openai" or "gemini"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Search Settings
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    COLLECTION_NAME: str = "technical_docs"
    
    # LangGraph Self-Correction Settings
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    
    # Web Search Fallback (Tavily or DuckDuckGo)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    ENABLE_WEB_SEARCH_FALLBACK: bool = True

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
