import logging
from app.graph.state import GraphState
from app.config import settings

logger = logging.getLogger(__name__)

def decide_after_grading(state: GraphState) -> str:
    """
    Conditional routing edge following Document Grading:
    1. If query is conversational recall -> proceed directly to 'generate'.
    2. If relevant documents exist -> proceed to 'generate'.
    3. If no relevant documents and retries remain -> route to 'rewrite_query'.
    4. If no relevant documents and retries exhausted -> route to 'fallback'.
    """
    if state.get("query_type") == "conversation_recall":
        logger.info("[Edge Router] Query is conversation recall -> routing directly to 'generate'")
        return "generate"

    filtered_docs = state.get("filtered_docs", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)
    
    if filtered_docs and len(filtered_docs) > 0:
        logger.info(f"[Edge Router] {len(filtered_docs)} relevant docs found -> routing to 'generate'")
        return "generate"
        
    if retry_count < max_retries:
        logger.info(f"[Edge Router] No relevant docs found. Retrying ({retry_count}/{max_retries}) -> routing to 'rewrite_query'")
        return "rewrite_query"
        
    logger.info("[Edge Router] No relevant docs found and max retries reached -> routing to 'fallback'")
    return "fallback"

def decide_after_hallucination(state: GraphState) -> str:
    """
    Evaluates groundedness after hallucination check:
    If not grounded and hallucination_retry <= 1 -> route back to 'generate'.
    Otherwise -> route to 'end'.
    """
    is_grounded = state.get("is_grounded", True)
    hallucination_retry = state.get("hallucination_retry", 0)
    
    if not is_grounded and hallucination_retry <= 1:
        logger.info(f"[Edge Router] Answer ungrounded (retry {hallucination_retry}/1). Routing back to 'generate'.")
        return "generate"
        
    logger.info("[Edge Router] Answer grounded or max regeneration reached -> routing to 'end'")
    return "end"
