
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split page documents into overlapping chunks for semantic retrieval."""

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index + 1

    return chunks
