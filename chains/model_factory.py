
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Create the configured chat model while keeping provider swaps isolated."""

    provider = settings.llm_provider

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini chat models.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            temperature=0.2,
            # Keep the model loaded between requests to avoid cold-start reloads.
            keep_alive=settings.ollama_keep_alive,
            # Cap generation length for QA; avoids runaway local decoding latency.
            num_predict=1024,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq chat models.")

        from langchain_groq import ChatGroq

        # Helpful for debugging: confirm which exact model id Groq is called with.
        import logging

        logging.getLogger(__name__).info(
            "Groq model init: model=%s provider=%s",
            settings.groq_model,
            settings.llm_provider,
        )

        return ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.2,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

