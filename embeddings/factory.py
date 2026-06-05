
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import Settings


def get_embedding_model(settings: Settings) -> Embeddings:
    """Create the configured embedding model."""

    provider = settings.embedding_provider

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings.")

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.ollama_embedding_model)

    raise ValueError(f"Unsupported embedding provider: {provider}")

