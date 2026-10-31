import json
import re
import logging
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from app.graph.state import GraphState
from app.services.llm_factory import get_chat_model
from app.services.vectorstore import search_documents
from app.services.search import web_search_fallback
from app.config import settings

logger = logging.getLogger(__name__)

def _extract_json(text: str) -> Dict[str, Any]:
    """Helper to safely extract JSON from LLM markdown code blocks or raw text."""
    try:
        # First attempt direct parse
        return json.loads(text.strip())
    except Exception:
        pass
    
    # Try regex match for ```json { ... } ``` or bare { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}

# -------------------------------------------------------------
# Node 1: Query Analysis & Classification
# -------------------------------------------------------------
def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user's raw question, classifies query intent,
    and expands it with technical terms for optimal vector retrieval.
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    system_prompt = (
        "You are an expert technical documentation assistant query analyzer.\n"
        "Analyze the user question and output a JSON object with two fields:\n"
        "1. 'query_type': One of ['conceptual', 'how-to', 'troubleshooting', 'api-reference']\n"
        "2. 'expanded_query': An expanded version of the query enriched with technical keywords, "
        "synonyms, and framework terms to maximize vector retrieval accuracy.\n\n"
        "Return ONLY the valid JSON object."
    )
    
    history_context = ""
    if chat_history:
        history_context = "\nRecent Conversation:\n" + "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-3:]]
        )
        
    human_prompt = f"User Question: {question}{history_context}"
    
    llm = get_chat_model(temperature=0.0)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    parsed = _extract_json(response.content)
    
    query_type = parsed.get("query_type", "how-to")
    expanded = parsed.get("expanded_query", question)
    
    logger.info(f"[Query Analysis] Type: {query_type} | Expanded Query: {expanded}")
    
    return {
        "query_type": query_type,
        "transformed_query": expanded,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", settings.MAX_RETRIES),
        "is_fallback": False,
        "is_grounded": True,
    }

# -------------------------------------------------------------
# Node 2: Retrieval
# -------------------------------------------------------------
def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Searches the persistent Chroma vector store for top-k document chunks
    matching the transformed query.
    """
    query = state.get("transformed_query") or state["question"]
    logger.info(f"[Retrieval] Searching vector store with query: '{query}'")
    
    docs = search_documents(query=query, k=settings.TOP_K)
    logger.info(f"[Retrieval] Retrieved {len(docs)} chunks from vector store.")
    
    return {
        "documents": docs,
    }

# -------------------------------------------------------------
# Node 3: Document Grading (Self-Correction)
# -------------------------------------------------------------
def document_grading_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates whether each retrieved document is truly relevant to the question.
    Filters out irrelevant chunks to prevent context pollution and hallucination.
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    system_prompt = (
        "You are a strict document grader assessing relevance of technical documentation chunks.\n"
        "Given the user question and a retrieved document chunk, grade whether the chunk contains "
        "information relevant to answering the question.\n"
        "Output ONLY a JSON object: {\"score\": \"relevant\"} or {\"score\": \"irrelevant\", \"reason\": \"...\"}."
    )
    
    llm = get_chat_model(temperature=0.0)
    filtered_docs: List[Document] = []
    
    for idx, doc in enumerate(documents):
        human_prompt = (
            f"User Question: {question}\n\n"
            f"Retrieved Document Chunk (Source: {doc.metadata.get('source')} | Section: {doc.metadata.get('section')}):\n"
            f"\"\"\"\n{doc.page_content}\n\"\"\""
        )
        
        try:
            res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            grade = _extract_json(res.content)
            score = grade.get("score", "irrelevant").strip().lower()
            if score == "relevant":
                filtered_docs.append(doc)
                logger.info(f"[Grader] Chunk {idx} ({doc.metadata.get('source')}) -> RELEVANT")
            else:
                logger.info(f"[Grader] Chunk {idx} ({doc.metadata.get('source')}) -> IRRELEVANT ({grade.get('reason', 'Not relevant')})")
        except Exception as e:
            logger.warning(f"[Grader] Error for chunk {idx}: {e}. Dropping chunk.")
            
    logger.info(f"[Grader Summary] {len(filtered_docs)}/{len(documents)} chunks deemed relevant.")
    return {
        "filtered_docs": filtered_docs
    }

# -------------------------------------------------------------
# Query Rewriter Node (Self-Correction Retry Loop)
# -------------------------------------------------------------
def query_rewriter_node(state: GraphState) -> Dict[str, Any]:
    """
    Rewrites the query into an alternate semantic representation when
    previous retrieval yielded no relevant documents. Increments retry counter.
    """
    current_retries = state.get("retry_count", 0) + 1
    question = state["question"]
    previous_query = state.get("transformed_query", question)
    
    system_prompt = (
        "You are an expert search query reformulator for technical documentation.\n"
        "The previous query failed to retrieve relevant documents. "
        "Formulate a completely rewritten search query that generalizes terms, uses alternative technical "
        "synonyms, and removes restrictive phrasing.\n"
        "Output ONLY the new query as plain text (no quotes, no markdown)."
    )
    
    human_prompt = f"Original Question: {question}\nPrevious Failed Query: {previous_query}"
    
    llm = get_chat_model(temperature=0.3)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    new_query = response.content.strip().strip('"')
    
    logger.info(f"[Query Rewriter] Retry {current_retries}/{state.get('max_retries', settings.MAX_RETRIES)}: '{new_query}'")
    
    return {
        "transformed_query": new_query,
        "retry_count": current_retries
    }

# -------------------------------------------------------------
# Web Search Fallback Node (Bonus)
# -------------------------------------------------------------
def web_search_fallback_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes an external web search fallback when local corpus does not contain
    the required information after maximum retries.
    """
    question = state["question"]
    logger.info(f"[Fallback] Vector store exhausted. Triggering web search for: '{question}'")
    
    web_docs = web_search_fallback(query=question, max_results=3)
    
    if web_docs:
        logger.info(f"[Fallback] Web search returned {len(web_docs)} external chunks.")
        return {
            "filtered_docs": web_docs,
            "is_fallback": True
        }
    else:
        logger.info("[Fallback] Web search returned 0 results. Setting empty context.")
        return {
            "filtered_docs": [],
            "is_fallback": True
        }

# -------------------------------------------------------------
# Node 4: Generation & Citations
# -------------------------------------------------------------
def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Generates an accurate, grounded answer using the validated context chunks.
    Constructs precise source citations.
    """
    question = state["question"]
    docs = state.get("filtered_docs", [])
    is_fallback = state.get("is_fallback", False)
    query_type = state.get("query_type", "how-to")
    
    # If no documents are available even after fallbacks
    if not docs:
        fallback_msg = (
            "I apologize, but I could not find sufficient information in the provided technical documentation "
            f"or available web search to answer your question: '{question}'.\n\n"
            "Please check if the topic is covered in your indexed documents, or try rephrasing your question."
        )
        current_history = list(state.get("chat_history") or [])
        current_history.append({"role": "user", "content": question})
        current_history.append({"role": "assistant", "content": fallback_msg})
        return {
            "generation": fallback_msg,
            "citations": [],
            "is_fallback": True,
            "is_grounded": True,
            "chat_history": current_history
        }
        
    # Format context with explicit citation references
    context_blocks = []
    citations: List[Dict[str, Any]] = []
    
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        sec = doc.metadata.get("section", "General")
        context_blocks.append(
            f"[Source {i}: {src} | Section: {sec}]\n{doc.page_content}"
        )
        citations.append({
            "source": src,
            "section": sec,
            "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })
        
    context_str = "\n\n---\n\n".join(context_blocks)
    
    hallucination_warning = ""
    if state.get("hallucination_retry", 0) > 0 and not state.get("is_grounded", True):
        hallucination_warning = (
            "\n6. CRITICAL REGENERATION: The previous answer contained ungrounded claims. "
            "Strictly restrict your answer to facts directly proven by the context excerpts."
        )

    system_prompt = (
        "You are an elite AI technical documentation assistant.\n"
        "Your task is to provide a clear, accurate, and comprehensive answer to the user's question, "
        "STRICTLY grounded in the provided documentation context.\n\n"
        "Rules:\n"
        "1. Rely ONLY on the provided context. Do NOT fabricate features or syntax.\n"
        "2. If the context contains code snippets, provide complete, syntactically correct examples.\n"
        "3. In your response, explicitly reference source documents using citations like [Source: filename, Section: name].\n"
        "4. Structure the response logically (e.g., Explanation, Code Example, Key Takeaways).\n"
        f"5. Tailor the tone to query type: {query_type}.{hallucination_warning}"
    )
    
    human_prompt = f"User Question: {question}\n\nDocumentation Context:\n{context_str}"
    
    llm = get_chat_model(temperature=0.1)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    
    current_history = list(state.get("chat_history") or [])
    current_history.append({"role": "user", "content": question})
    current_history.append({"role": "assistant", "content": response.content})

    return {
        "generation": response.content,
        "citations": citations,
        "is_fallback": is_fallback,
        "chat_history": current_history
    }

# -------------------------------------------------------------
# Bonus Node: Hallucination / Groundedness Check (Self-RAG)
# -------------------------------------------------------------
def hallucination_check_node(state: GraphState) -> Dict[str, Any]:
    """
    Self-RAG inspired verification: checks whether the generated answer is strictly
    grounded in the retrieved context documents.
    """
    generation = state.get("generation", "")
    docs = state.get("filtered_docs", [])
    current_retry = state.get("hallucination_retry", 0)
    
    if not docs or state.get("is_fallback", False):
        return {"is_grounded": True, "hallucination_retry": current_retry}
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    system_prompt = (
        "You are a strict hallucination checker evaluating an AI-generated technical answer.\n"
        "Check whether every factual assertion and code pattern in the answer is supported by the context.\n"
        "Output ONLY a JSON: {\"grounded\": \"yes\"} or {\"grounded\": \"no\", \"explanation\": \"...\"}."
    )
    
    human_prompt = f"Context:\n{context_text}\n\nAnswer to check:\n{generation}"
    
    llm = get_chat_model(temperature=0.0)
    try:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        parsed = _extract_json(res.content)
        grounded = parsed.get("grounded", "yes").lower() == "yes"
        new_retry = current_retry + (1 if not grounded else 0)
        logger.info(f"[Hallucination Check] Grounded: {grounded} | Retries: {new_retry}")
        return {"is_grounded": grounded, "hallucination_retry": new_retry}
    except Exception as e:
        logger.warning(f"[Hallucination Check] Error: {e}. Assuming grounded.")
        return {"is_grounded": True, "hallucination_retry": current_retry}
