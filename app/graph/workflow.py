import logging
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import GraphState
from app.graph.nodes import (
    query_analysis_node,
    retrieval_node,
    document_grading_node,
    query_rewriter_node,
    web_search_fallback_node,
    generation_node,
    hallucination_check_node,
)
from app.graph.edges import decide_after_grading, decide_after_hallucination
from app.config import settings

logger = logging.getLogger(__name__)

def build_graph() -> StateGraph:
    """
    Constructs and connects the self-corrective LangGraph StateGraph.
    Includes query analysis, vector retrieval, document grading,
    retry query reformulation loop, web search fallback, citation generation,
    and hallucination verification with single-retry regeneration.
    """
    workflow = StateGraph(GraphState)
    
    # 1. Add Nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade_documents", document_grading_node)
    workflow.add_node("rewrite_query", query_rewriter_node)
    workflow.add_node("fallback", web_search_fallback_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    
    # 2. Add Edges
    workflow.set_entry_point("query_analysis")
    
    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # 3. Conditional Edge based on document grading outcome
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        }
    )
    
    # Retry loop: rewrote query -> re-retrieve
    workflow.add_edge("rewrite_query", "retrieve")
    
    # Fallback to generation
    workflow.add_edge("fallback", "generate")
    
    # Groundedness / Hallucination verification
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {
            "generate": "generate",
            "end": END,
        }
    )
    
    return workflow

# Compile graph with MemorySaver checkpointer for conversational thread support
_checkpointer = MemorySaver()
compiled_app = build_graph().compile(checkpointer=_checkpointer)

async def run_documentation_assistant(
    question: str, 
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Asynchronously invokes the LangGraph pipeline for a user question.
    Thread ID manages conversation history and session isolation.
    Notice chat_history is NOT reset to [] in initial_state, allowing
    MemorySaver to preserve conversation history across turns.
    """
    tid = thread_id or "default_session"
    config = {"configurable": {"thread_id": tid}}
    
    # Notice: 'chat_history' is purposefully omitted from input_state
    # so MemorySaver preserves multi-turn conversation memory across calls.
    input_state = {
        "question": question,
        "query_type": "how-to",
        "transformed_query": question,
        "documents": [],
        "filtered_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": settings.MAX_RETRIES,
        "is_fallback": False,
        "is_grounded": True,
        "hallucination_retry": 0,
    }
    
    final_state = await compiled_app.ainvoke(input_state, config=config)
    return final_state
