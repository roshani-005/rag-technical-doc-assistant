import os
import logging
from typing import List
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

def web_search_fallback(query: str, max_results: int = 3) -> List[Document]:
    """
    Executes a web search fallback when vector store retrieved zero relevant documents.
    Attempts Tavily Search API if key is present, otherwise falls back to DuckDuckGo search.
    """
    tavily_key = settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
    
    # 1. Tavily API Search
    if tavily_key:
        try:
            import httpx
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": max_results,
            }
            resp = httpx.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                docs = []
                for item in results:
                    docs.append(
                        Document(
                            page_content=item.get("content", ""),
                            metadata={
                                "source": item.get("url", "web_search"),
                                "section": item.get("title", "Web Result"),
                                "chunk_id": f"web_{item.get('url', '')[:30]}"
                            }
                        )
                    )
                if docs:
                    return docs
        except Exception as e:
            logger.warning(f"Tavily web search error: {e}. Trying DuckDuckGo...")

    # 2. DuckDuckGo Free Search Fallback
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r.get("body", ""),
                    metadata={
                        "source": r.get("href", "duckduckgo"),
                        "section": r.get("title", "Web Result"),
                        "chunk_id": f"ddg_{r.get('href', '')[:30]}"
                    }
                )
            )
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}.")

    # 3. Empty list if unavailable
    return []
