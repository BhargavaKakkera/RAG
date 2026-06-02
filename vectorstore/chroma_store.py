from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings


def build_chroma_store(
    documents: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    collection_name: str,
) -> Chroma:
    """Create a Chroma collection from chunked documents."""

    if not documents:
        raise ValueError("Cannot build a vector store without documents.")

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

