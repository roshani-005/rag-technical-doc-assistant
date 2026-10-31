#!/usr/bin/env python3
"""
Standalone Document Ingestion CLI for RAG Technical Documentation Assistant.
Loads technical documentation (Markdown, text, HTML, or URLs), applies hierarchical
Markdown-aware chunking, generates embeddings, and persists them into ChromaDB.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.ingestion import ingest_directory, ingest_urls
from app.services.vectorstore import get_corpus_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest-cli")

def main():
    parser = argparse.ArgumentParser(
        description="Ingest technical documentation into Chroma vector store."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(settings.DOCS_DIR),
        help=f"Directory containing documentation files (default: {settings.DOCS_DIR})"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="One or more remote documentation URLs to scrape and index."
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print current vector store index statistics without ingesting new files."
    )
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_corpus_statistics()
        print("\n=== Corpus Index Statistics ===")
        print(f"Total Documents : {stats['total_documents']}")
        print(f"Total Chunks    : {stats['total_chunks']}")
        print("\nIndexed Sources:")
        for doc in stats["documents"]:
            print(f"- {doc['source']} ({doc['chunk_count']} chunks)")
            if doc['sections']:
                print(f"  Sections: {', '.join(doc['sections'][:5])}")
        return

    print("=" * 60)
    print(" RAG Technical Documentation Assistant - Document Ingestion")
    print("=" * 60)
    
    # Ingest URLs if provided
    if args.urls:
        print(f"[*] Ingesting {len(args.urls)} remote URLs...")
        res = ingest_urls(args.urls)
        print(f"[OK] Added {res['total_chunks_added']} chunks from {len(res['successful_urls'])} URLs.")
        
    # Ingest local directory
    target_dir = Path(args.dir)
    if target_dir.exists() and target_dir.is_dir():
        print(f"[*] Ingesting local documentation from: {target_dir}")
        res = ingest_directory(target_dir)
        print(f"[OK] Indexed {res['total_files']} files into {res['total_chunks_added']} vector chunks.")
    else:
        print(f"[!] Warning: Target directory '{target_dir}' does not exist.")
        
    print("\n[OK] Ingestion complete! Run 'python ingest.py --stats' to view current corpus.")

if __name__ == "__main__":
    main()
