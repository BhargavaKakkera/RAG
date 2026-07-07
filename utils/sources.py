
from __future__ import annotations

from langchain_core.documents import Document


def format_documents_for_prompt(
    documents: list[Document],
    max_chars: int | None = None,
) -> str:
    formatted_chunks: list[str] = []
    used_chars = 0

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "N/A")
        chunk_id = document.metadata.get("chunk_id", index)
        chunk_text = (
            f"[Chunk {index} | Source: {source} | Page: {page} | ID: {chunk_id}]\n"
            f"{document.page_content}"
        )

        if max_chars is not None and formatted_chunks:
            separator_len = 2  # "\n\n"
            if used_chars + separator_len + len(chunk_text) > max_chars:
                break

        if max_chars is not None and not formatted_chunks and len(chunk_text) > max_chars:
            chunk_text = chunk_text[:max_chars]

        formatted_chunks.append(chunk_text)
        used_chars += len(chunk_text)

    return "\n\n".join(formatted_chunks)


def source_rows(documents: list[Document]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, document in enumerate(documents, start=1):
        rows.append(
            {
                "rank": index,
                "source": document.metadata.get("source", "Unknown source"),
                "page": document.metadata.get("page", "N/A"),
                "document_id": document.metadata.get("document_id", "N/A"),
                "chunk_id": document.metadata.get("chunk_id", "N/A"),
                "similarity_score": document.metadata.get("similarity_score", "N/A"),
            }
        )
    return rows

