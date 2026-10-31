import os
import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.config import settings

logger = logging.getLogger(__name__)

_vectorstore_instance: Chroma | None = None

def get_embeddings():
    """
    Returns embedding model based on settings.
    Falls back gracefully if dependencies or API keys are missing.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    
    if provider == "openai" and (settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small"
            )
        except Exception as e:
            logger.warning(f"Failed to load OpenAIEmbeddings: {e}. Falling back to HuggingFace.")

    # Default: Local HuggingFace sentence-transformers (offline, free, fast)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        logger.warning(f"HuggingFaceEmbeddings not available ({e}). Using deterministic LocalFallbackEmbeddings.")
        
        # Super lightweight fallback embedding class for testing without downloads
        from langchain_core.embeddings import Embeddings
        import hashlib
        
        class LocalFallbackEmbeddings(Embeddings):
            def __init__(self, dim: int = 384):
                self.dim = dim
                
            def _embed(self, text: str) -> List[float]:
                # Generate a repeatable pseudo-embedding from text tokens
                words = text.lower().split()
                vec = [0.0] * self.dim
                for i, word in enumerate(words):
                    idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                    vec[idx] += 1.0 / (i + 1)
                norm = sum(x**2 for x in vec) ** 0.5
                return [x / norm if norm > 0 else 0.0 for x in vec]
                
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self._embed(t) for t in texts]
                
            def embed_query(self, text: str) -> List[float]:
                return self._embed(text)
                
        return LocalFallbackEmbeddings()

def get_vectorstore() -> Chroma:
    """
    Initializes and returns the persistent Chroma vector store.
    """
    global _vectorstore_instance
    if _vectorstore_instance is None:
        embeddings = get_embeddings()
        _vectorstore_instance = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(settings.CHROMA_DIR),
        )
    return _vectorstore_instance

def add_documents_to_store(docs: List[Document]) -> int:
    """
    Adds a list of Document objects to the Chroma vector store.
    """
    if not docs:
        return 0
    store = get_vectorstore()
    store.add_documents(docs)
    # Chroma >= 0.4.x automatically persists to disk
    try:
        if hasattr(store, "persist"):
            store.persist()
    except Exception:
        pass
    return len(docs)

def search_documents(query: str, k: int | None = None) -> List[Document]:
    """
    Performs similarity search in the vector store.
    """
    top_k = k or settings.TOP_K
    store = get_vectorstore()
    return store.similarity_search(query, k=top_k)

def search_documents_with_scores(query: str, k: int | None = None) -> List[Tuple[Document, float]]:
    """
    Performs similarity search returning documents with relevance/distance scores.
    """
    top_k = k or settings.TOP_K
    store = get_vectorstore()
    return store.similarity_search_with_score(query, k=top_k)

def get_corpus_statistics() -> Dict[str, Any]:
    """
    Returns metadata regarding all indexed documents and chunks.
    """
    store = get_vectorstore()
    try:
        data = store.get()
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        
        doc_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "unknown")
            section = meta.get("section", "General")
            if src not in doc_map:
                doc_map[src] = {"chunk_count": 0, "sections": set()}
            doc_map[src]["chunk_count"] += 1
            if section:
                doc_map[src]["sections"].add(section)
                
        doc_list = [
            {
                "source": src,
                "chunk_count": val["chunk_count"],
                "sections": sorted(list(val["sections"]))
            }
            for src, val in doc_map.items()
        ]
        
        return {
            "total_documents": len(doc_map),
            "total_chunks": len(ids),
            "documents": doc_list
        }
    except Exception as e:
        logger.error(f"Error fetching corpus stats: {e}")
        return {"total_documents": 0, "total_chunks": 0, "documents": []}
