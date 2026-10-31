import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from app.config import settings
from app.services.vectorstore import add_documents_to_store

logger = logging.getLogger(__name__)

# Headers to split on for Markdown structure preservation
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def get_deterministic_chunk_id(source_name: str, content: str) -> str:
    """
    Computes a deterministic chunk ID based on the source name and content hash.
    Ensures re-ingesting the same file updates records idempotently rather than duplicating.
    """
    digest = hashlib.sha256(f"{source_name}:{content.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"{source_name}_{digest}"

def load_text_from_file(file_path: Path) -> str:
    """Reads raw content from a file with utf-8 fallback."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def fetch_content_from_url(url: str) -> str:
    """
    Fetches web page HTML and parses readable text/markdown from main article containers.
    """
    headers = {"User-Agent": "RAG-Documentation-Assistant/1.0"}
    response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove script, style, nav, footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
        
    # Prefer main or article tags if present
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    text = main_content.get_text(separator="\n", strip=True) if main_content else soup.get_text(strip=True)
    return text

def chunk_markdown_document(content: str, source_name: str) -> List[Document]:
    """
    Hierarchical 2-stage chunking strategy for technical markdown:
    1. Splits by markdown headers to keep conceptual sections intact.
    2. Recursively splits large sections by code blocks and paragraphs.
    """
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )
    header_splits = markdown_splitter.split_text(content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n```", "```\n", "\n\n", "\n", " ", ""]
    )
    
    final_chunks: List[Document] = []
    for split in header_splits:
        sub_docs = text_splitter.split_documents([split])
        for idx, doc in enumerate(sub_docs):
            headers = [
                doc.metadata.get("Header 1"),
                doc.metadata.get("Header 2"),
                doc.metadata.get("Header 3")
            ]
            section = " > ".join([h for h in headers if h]) or "General"
            chunk_id = get_deterministic_chunk_id(source_name, doc.page_content)
            
            clean_metadata = {
                "source": source_name,
                "section": section,
                "chunk_id": chunk_id
            }
            final_chunks.append(Document(page_content=doc.page_content, metadata=clean_metadata))
            
    # If document had no markdown headers, fall back to recursive splitter
    if not final_chunks:
        sub_docs = text_splitter.split_text(content)
        for chunk in sub_docs:
            chunk_id = get_deterministic_chunk_id(source_name, chunk)
            final_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source_name, "section": "General", "chunk_id": chunk_id}
                )
            )
        
    return final_chunks

def chunk_plain_document(content: str, source_name: str) -> List[Document]:
    """Splits plain text or HTML extracted text with deterministic chunk IDs."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = text_splitter.split_text(content)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "section": "General",
                "chunk_id": get_deterministic_chunk_id(source_name, chunk)
            }
        )
        for chunk in raw_chunks
    ]
    return documents

def clean_html_content(raw_html: str) -> str:
    """Strips HTML tags, scripts, and navigation elements from HTML documents."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return main.get_text(separator="\n", strip=True)

def ingest_directory(directory_path: Path) -> Dict[str, Any]:
    """
    Ingests all markdown, text, and html documents from a directory into the vector store.
    """
    supported_extensions = {".md", ".txt", ".html", ".htm"}
    all_chunks: List[Document] = []
    indexed_files: List[str] = []
    
    for file_path in directory_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            content = load_text_from_file(file_path)
            ext = file_path.suffix.lower()
            if ext == ".md":
                chunks = chunk_markdown_document(content, file_path.name)
            elif ext in (".html", ".htm"):
                text = clean_html_content(content)
                chunks = chunk_plain_document(text, file_path.name)
            else:
                chunks = chunk_plain_document(content, file_path.name)
                
            all_chunks.extend(chunks)
            indexed_files.append(file_path.name)
            
    count = add_documents_to_store(all_chunks)
    return {
        "status": "success",
        "total_files": len(indexed_files),
        "files": indexed_files,
        "total_chunks_added": count
    }

def ingest_single_file(file_content: str, filename: str) -> int:
    """Ingests a single uploaded file, parsing HTML if applicable."""
    ext = filename.lower()
    if ext.endswith(".md"):
        chunks = chunk_markdown_document(file_content, filename)
    elif ext.endswith(".html") or ext.endswith(".htm"):
        text = clean_html_content(file_content)
        chunks = chunk_plain_document(text, filename)
    else:
        chunks = chunk_plain_document(file_content, filename)
    return add_documents_to_store(chunks)

def ingest_urls(urls: List[str]) -> Dict[str, Any]:
    """Fetches, chunks, and indexes a list of remote documentation URLs."""
    total_added = 0
    successful_urls = []
    for url in urls:
        try:
            content = fetch_content_from_url(url)
            clean_name = re.sub(r"[^\w\-_\.]", "_", url.replace("https://", "").replace("http://", ""))
            chunks = chunk_plain_document(content, clean_name[:60])
            count = add_documents_to_store(chunks)
            total_added += count
            successful_urls.append(url)
        except Exception as e:
            logger.error(f"Failed to ingest URL {url}: {e}")
            
    return {
        "status": "success" if successful_urls else "failed",
        "successful_urls": successful_urls,
        "total_chunks_added": total_added
    }
