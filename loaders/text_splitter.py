
from __future__ import annotations

from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    document_ids: dict[str, str] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in document_ids:
            document_ids[source] = sha256(source.encode("utf-8")).hexdigest()[:16]

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["document_id"] = document_ids[source]
        chunk.metadata["chunk_id"] = index + 1

    return chunks
