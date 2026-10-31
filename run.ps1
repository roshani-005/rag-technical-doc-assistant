# Windows PowerShell Launcher for RAG Technical Documentation Assistant

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting RAG Technical Documentation Assistant Backend" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VENV_PYTHON = ".\.venv\Scripts\python.exe"

if (-Not (Test-Path $VENV_PYTHON)) {
    Write-Host "[!] Virtual environment not found. Setting up..." -ForegroundColor Yellow
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

# Run initial ingestion if vector DB not populated
Write-Host "[*] Ingesting technical documentation corpus..." -ForegroundColor Green
& $VENV_PYTHON ingest.py

Write-Host "`n[✓] Starting FastAPI server on http://localhost:8000..." -ForegroundColor Green
Write-Host "    Swagger Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "    (To launch Streamlit UI in another terminal: .\.venv\Scripts\streamlit.exe run frontend.py)`n" -ForegroundColor Yellow

& $VENV_PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
