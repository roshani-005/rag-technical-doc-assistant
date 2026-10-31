import pytest
from langchain_core.documents import Document
from app.graph.state import GraphState
from app.graph.edges import decide_after_grading
from app.graph.nodes import query_analysis_node, document_grading_node, query_rewriter_node

def test_decide_after_grading_with_relevant_docs():
    state: GraphState = {
        "question": "How to declare query params?",
        "query_type": "how-to",
        "transformed_query": "query params",
        "documents": [],
        "filtered_docs": [Document(page_content="Use Query()", metadata={"source": "test.md"})],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
        "is_grounded": True,
        "chat_history": []
    }
    decision = decide_after_grading(state)
    assert decision == "generate"

def test_decide_after_grading_with_zero_docs_under_retry_limit():
    state: GraphState = {
        "question": "Unknown concept?",
        "query_type": "conceptual",
        "transformed_query": "unknown concept",
        "documents": [],
        "filtered_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
        "is_grounded": True,
        "chat_history": []
    }
    decision = decide_after_grading(state)
    assert decision == "rewrite_query"

def test_decide_after_grading_with_retries_exhausted():
    state: GraphState = {
        "question": "Nonexistent topic?",
        "query_type": "conceptual",
        "transformed_query": "nonexistent topic",
        "documents": [],
        "filtered_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 2,
        "max_retries": 2,
        "is_fallback": False,
        "is_grounded": True,
        "chat_history": []
    }
    decision = decide_after_grading(state)
    assert decision == "fallback"

def test_query_analysis_node():
    state: GraphState = {
        "question": "How to validate request body with Pydantic in FastAPI?",
        "query_type": "",
        "transformed_query": "",
        "documents": [],
        "filtered_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
        "is_grounded": True,
        "chat_history": []
    }
    result = query_analysis_node(state)
    assert "query_type" in result
    assert "transformed_query" in result
    assert len(result["transformed_query"]) > 0

def test_query_rewriter_increments_retry_count():
    state: GraphState = {
        "question": "How do I do dependency injection?",
        "query_type": "how-to",
        "transformed_query": "dependency injection",
        "documents": [],
        "filtered_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 1,
        "max_retries": 2,
        "is_fallback": False,
        "is_grounded": True,
        "chat_history": []
    }
    result = query_rewriter_node(state)
    assert result["retry_count"] == 2
    assert "transformed_query" in result
