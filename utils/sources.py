
from __future__ import annotations

from langchain_core.documents import Document


def format_documents_for_prompt(documents: list[Document]) -> str:
    """Render retrieved chunks into a compact context block."""

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "N/A")
        chunk_id = document.metadata.get("chunk_id", index)
        formatted_chunks.append(
            f"[Chunk {index} | Source: {source} | Page: {page} | ID: {chunk_id}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def source_rows(documents: list[Document]) -> list[dict[str, object]]:
    """Build display-friendly source metadata for Streamlit."""

    rows: list[dict[str, object]] = []
    for index, document in enumerate(documents, start=1):
        rows.append(
            {
                "rank": index,
                "source": document.metadata.get("source", "Unknown source"),
                "page": document.metadata.get("page", "N/A"),
                "chunk_id": document.metadata.get("chunk_id", "N/A"),
                "similarity_score": document.metadata.get("similarity_score", "N/A"),
            }
        )
    return rows

