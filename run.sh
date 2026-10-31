#!/usr/bin/env bash
# Linux / macOS Launcher for RAG Technical Documentation Assistant

set -e

echo "=========================================================="
echo "  Starting RAG Technical Documentation Assistant Backend"
echo "=========================================================="

if [ ! -d ".venv" ]; then
    echo "[!] Setting up Python virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

echo "[*] Ingesting technical documentation corpus..."
.venv/bin/python ingest.py

echo ""
echo "[✓] Starting FastAPI server on http://localhost:8000..."
echo "    Swagger Docs: http://localhost:8000/docs"
echo "    (To launch Streamlit UI in another terminal: .venv/bin/streamlit run frontend.py)"
echo ""

.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
