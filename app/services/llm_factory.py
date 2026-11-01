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
    Deterministic offline fallback model for unit tests and local demonstrations
    when LLM_PROVIDER is explicitly configured as 'mock'.
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
        if "grade whether the chunk contains" in last_lower or "is the document relevant" in last_lower:
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
        # Query rewriter node
        elif "reformulate" in last_lower or "rewritten search query" in last_lower:
            response_text = "FastAPI query parameters optional defaults"
        # Hallucination check
        elif "hallucination" in last_lower or "grounded" in last_lower:
            response_text = '{"grounded": "yes", "explanation": "Every statement is directly supported by the provided context."}'
        # Conversation history recall
        elif "conversation history" in last_lower or "discussed" in last_lower or "studied above" in last_lower:
            response_text = "Earlier in our conversation, we discussed FastAPI Overview, Path Parameters, and Request Bodies."
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

def get_chat_model(temperature: float = 0.0, max_tokens: Optional[int] = None) -> BaseChatModel:
    """
    Factory function returning the configured ChatModel.
    Supports Groq, OpenAI, Google Gemini, and MockChatModel.
    Raises ValueError if API keys are missing for non-mock providers.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "mock":
        return MockChatModel()
        
    elif provider == "groq":
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please set it in your environment or .env file, "
                "or set LLM_PROVIDER=mock to use the offline mock model."
            )
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=settings.GROQ_MODEL,
                groq_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens or 400,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {e}")
            raise

    elif provider in ("gemini", "google"):
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set it in your environment or .env file, "
                "or set LLM_PROVIDER=mock to use the offline mock model."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise

    elif provider == "openai":
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Please set it in your environment or .env file, "
                "or set LLM_PROVIDER=mock to use the offline mock model."
            )
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Supported providers are: 'groq', 'gemini', 'openai', 'mock'."
        )
