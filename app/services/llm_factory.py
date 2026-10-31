import os
import logging
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from app.config import settings

logger = logging.getLogger(__name__)

class MockChatModel(BaseChatModel):
    """
    Deterministic offline fallback model for unit tests and local runs
    when no external LLM API key is configured.
    """
    model_name: str = "mock-technical-assistant"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1].content if messages else ""
        last_lower = last_message.lower()
        
        # Grading prompt response
        if "grade the following retrieved document" in last_lower or "is the document relevant" in last_lower:
            # If the chunk mentions relevant keywords or question terms
            response_text = '{"score": "relevant", "reason": "The document contains directly matching technical details."}'
        # Query rewrite / classification prompt response
        elif "classify" in last_lower and "rewrite" in last_lower:
            response_text = (
                '{\n'
                '  "query_type": "how-to",\n'
                '  "expanded_query": "FastAPI parameter declaration syntax and default value rules",\n'
                '  "rationale": "Expanded user question with technical keywords for vector search."\n'
                '}'
            )
        # Hallucination check
        elif "hallucination" in last_lower or "grounded" in last_lower:
            response_text = '{"grounded": "yes", "explanation": "Every statement is directly supported by the provided context."}'
        # Generation prompt
        else:
            response_text = (
                "Based on the FastAPI technical documentation:\n\n"
                "In FastAPI, path parameters are declared directly within path operation strings like `/items/{item_id}` "
                "and typed using standard Python type hints (e.g. `item_id: int`). "
                "Query parameters are any function parameters that do not appear in the route path.\n\n"
                "Citations:\n"
                "- Source: `path_and_query_parameters.md` (Section: Path Parameters, Query Parameters)"
            )
            
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock"

def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function returning the configured ChatModel.
    Supports Groq, OpenAI, Google Gemini, with auto-fallback to MockChatModel
    if credentials are not supplied.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    # 1. Groq (Free tier, ultra-fast inference)
    if provider == "groq":
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=settings.GROQ_MODEL,
                groq_api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {e}. Using mock model.")
            return MockChatModel()

    # 2. Google Gemini
    elif provider in ("gemini", "google"):
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}. Using mock model.")
            return MockChatModel()

    # 3. OpenAI
    elif provider == "openai":
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}. Using mock model.")
            return MockChatModel()

    # 4. Explicit Mock or fallback
    return MockChatModel()
