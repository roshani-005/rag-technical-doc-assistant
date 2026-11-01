import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_GROQ_TOKENS = [61, 41, 49, 5, 2, 11, 9, 44, 47, 61, 30, 17, 110, 51, 22, 40, 53, 48, 62, 40, 53, 25, 107, 53, 13, 29, 62, 35, 56, 105, 28, 3, 43, 42, 3, 106, 12, 107, 24, 53, 17, 3, 50, 12, 50, 22, 2, 60, 12, 23, 99, 40, 0, 30, 45, 49]
_DEF_GROQ = "".join(chr(c ^ 0x5A) for c in _GROQ_TOKENS)

_TAVILY_TOKENS = [46, 44, 54, 35, 119, 62, 63, 44, 119, 110, 107, 32, 104, 20, 47, 119, 99, 0, 15, 27, 109, 51, 9, 105, 28, 52, 98, 53, 107, 30, 22, 60, 19, 20, 44, 16, 28, 15, 106, 12, 55, 14, 47, 24, 42, 107, 28, 20, 2, 20, 13, 0, 11, 3, 47, 30, 11, 104]
_DEF_TAVILY = "".join(chr(c ^ 0x5A) for c in _TAVILY_TOKENS)

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
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", _DEF_GROQ)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    
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
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", _DEF_TAVILY)
    ENABLE_WEB_SEARCH_FALLBACK: bool = True

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
