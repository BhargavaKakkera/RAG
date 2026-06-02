from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Central runtime settings loaded from environment variables."""

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
    )

    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    huggingface_embedding_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    default_chunk_size: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    default_chunk_overlap: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "150"))
    default_retrieval_k: int = int(os.getenv("DEFAULT_RETRIEVAL_K", "5"))
    default_similarity_threshold: float = float(
        os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.35")
    )
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))


settings = Settings()

