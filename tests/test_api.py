import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "endpoints" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks" in data

def test_list_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert isinstance(data["documents"], list)

def test_feedback_submission():
    payload = {
        "query_id": "test_q_123",
        "rating": "up",
        "comment": "Accurate and clear citation!",
        "question": "What is an ASGI server?"
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data

def test_query_validation_error_on_empty_string():
    payload = {"question": "   "}
    response = client.post("/query", json=payload)
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower()
