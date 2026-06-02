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

        return ChatOllama(model=settings.ollama_model, temperature=0.2)

    raise ValueError(f"Unsupported LLM provider: {provider}")

