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
    
    # Detect if user is asking about the previous conversation
    q_lower = question.lower()
    is_recall = bool(
        chat_history and any(
            phrase in q_lower for phrase in [
                "studied above", "discussed above", "we discuss", "we study", "we studied",
                "what did we", "earlier", "previous answer", "previous question", "summarize what we",
                "what was my last", "what were we talking"
            ]
        )
    )

    system_prompt = (
        "You are an expert technical documentation assistant query analyzer.\n"
        "Analyze the user question in the context of recent conversation history.\n"
        "Output a JSON object with two fields:\n"
        "1. 'query_type': One of ['conceptual', 'how-to', 'troubleshooting', 'api-reference', 'conversation_recall']\n"
        "   - Use 'conversation_recall' if the user asks what was discussed or studied earlier in the conversation.\n"
        "2. 'expanded_query': A self-contained, grammatically corrected search query. "
        "Correct any obvious technical typos (e.g., 'fastpi' -> 'FastAPI', 'pydanticc' -> 'Pydantic'). "
        "Resolve any pronouns ('it', 'that', 'these') from recent conversation.\n\n"
        "Return ONLY the valid JSON object."
    )
    
    history_context = ""
    if chat_history:
        history_context = "\nRecent Conversation:\n" + "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-4:]]
        )
        
    human_prompt = f"User Question: {question}{history_context}"
    
    llm = get_chat_model(temperature=0.0, max_tokens=100)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        parsed = _extract_json(response.content)
        query_type = parsed.get("query_type", "conversation_recall" if is_recall else "how-to")
        expanded = parsed.get("expanded_query", question)
    except Exception as e:
        logger.warning(f"[Query Analysis] LLM error: {e}. Using raw question.")
        query_type = "conversation_recall" if is_recall else "how-to"
        expanded = question
        
    if is_recall:
        query_type = "conversation_recall"
    
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
    
    llm = get_chat_model(temperature=0.0, max_tokens=50)
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
    
    llm = get_chat_model(temperature=0.3, max_tokens=50)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        new_query = response.content.strip().strip('"')
    except Exception as e:
        logger.warning(f"[Query Rewriter] Rewriter error: {e}. Using fallback query.")
        new_query = f"{question} technical details"
    
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
    
    chat_history = list(state.get("chat_history") or [])
    
    # 1. Handle conversation recall (asking what was discussed/studied earlier in the session)
    if query_type == "conversation_recall" or (chat_history and any(
        phrase in question.lower() for phrase in [
            "studied above", "discussed above", "we discuss", "we study", "we studied",
            "what did we", "earlier in our", "previous answer", "previous question", "summarize what we"
        ]
    )):
        history_lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_text = "\n\n".join(history_lines)
        
        system_prompt = (
            "You are an AI technical documentation assistant with full memory of this session.\n"
            "The user is asking what was discussed, studied, or answered earlier in the conversation.\n"
            "Review the conversation history and clearly summarize the topics and key takeaways.\n"
            "Be polite, clear, and structure your answer with bullet points."
        )
        human_prompt = f"Conversation History:\n{history_text}\n\nUser Question: {question}"
        
        llm = get_chat_model(temperature=0.2, max_tokens=350)
        try:
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            answer_text = response.content
        except Exception as e:
            logger.warning(f"[Generation] Memory recall error: {e}")
            answer_text = "Earlier in our conversation, we discussed topics from the indexed technical documentation."
            
        current_history = list(chat_history)
        current_history.append({"role": "user", "content": question})
        current_history.append({"role": "assistant", "content": answer_text})
        return {
            "generation": answer_text,
            "citations": [],
            "is_fallback": False,
            "is_grounded": True,
            "chat_history": current_history
        }

    # 2. If no documents are available even after fallbacks
    if not docs:
        fallback_msg = (
            "I apologize, but I could not find sufficient information in the provided technical documentation "
            f"to answer your question: '{question}'.\n\n"
            "The indexed documentation covers: **FastAPI Overview, Path & Query Parameters, Request Body & Pydantic, Dependency Injection, and Error Handling**."
        )
        current_history = list(chat_history)
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
        "2. Tolerate minor user spelling typos or informal casing (e.g., 'fastpi' -> 'FastAPI', 'pydanticc' -> 'Pydantic'). "
        "Answer the core technical question directly using the provided context.\n"
        "3. If the context does not contain the answer, state clearly: "
        f"'The provided documentation does not contain information to answer: {question}.'\n"
        "4. In your response, explicitly reference source documents using citations like [Source: filename, Section: name].\n"
        "5. Structure the response logically (e.g., Explanation, Code Example, Key Takeaways).\n"
        f"6. Tailor the tone to query type: {query_type}.{hallucination_warning}"
    )
    
    history_str = ""
    if chat_history:
        history_lines = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in chat_history[-4:]]
        history_str = "Recent Conversation History (use this to resolve pronouns and follow-up context):\n" + "\n".join(history_lines) + "\n\n"
        
    human_prompt = f"{history_str}User Question: {question}\n\nDocumentation Context:\n{context_str}"
    
    llm = get_chat_model(temperature=0.1, max_tokens=350)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        answer_text = response.content
    except Exception as e:
        logger.warning(f"[Generation] LLM invocation failed: {e}. Returning safe not-found message.")
        answer_text = (
            f"The provided technical documentation does not contain information to answer: '{question}'.\n\n"
            "The indexed documentation covers: **FastAPI Overview, Path & Query Parameters, Request Body & Pydantic, Dependency Injection, and Error Handling**."
        )
        
    if is_fallback and not answer_text.startswith("> **[Note]**"):
        answer_text = f"> **[Note]** This topic was not found in the local technical documentation.\n\n{answer_text}"
    
    current_history = list(state.get("chat_history") or [])
    current_history.append({"role": "user", "content": question})
    current_history.append({"role": "assistant", "content": answer_text})

    return {
        "generation": answer_text,
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
    
    llm = get_chat_model(temperature=0.0, max_tokens=50)
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
