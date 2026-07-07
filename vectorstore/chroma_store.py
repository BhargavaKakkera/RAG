from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings
from utils.timing import log_timing

COLLECTION_PREFIX = "pdf_rag_"
COSINE_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def compute_upload_fingerprint(
    uploaded_files,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    """SHA256 fingerprint over PDF bytes, including chunking parameters so changing them triggers re-indexing."""

    sorted_files = sorted(uploaded_files, key=lambda f: f.name)
    digest = sha256()

    # Include chunking config (if provided) so embeddings are consistent.
    digest.update(f"chunk_size={chunk_size}".encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"chunk_overlap={chunk_overlap}".encode("utf-8"))
    digest.update(b"\0")

    for uploaded_file in sorted_files:
        try:
            content = uploaded_file.getvalue()
        except Exception:
            uploaded_file.seek(0)
            content = uploaded_file.read()
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")

    return digest.hexdigest()



def collection_name_for_fingerprint(fingerprint: str) -> str:
    return f"{COLLECTION_PREFIX}{fingerprint}"


def _metadata_path(settings: Settings, fingerprint: str) -> Path:
    metadata_dir = Path(settings.chroma_persist_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f"{fingerprint}.json"


def load_document_metadata(settings: Settings, fingerprint: str) -> dict | None:
    path = _metadata_path(settings, fingerprint)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_document_metadata(
    settings: Settings,
    fingerprint: str,
    metadata: dict,
) -> None:
    path = _metadata_path(settings, fingerprint)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def collection_exists(settings: Settings, collection_name: str) -> bool:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    existing = {collection.name for collection in client.list_collections()}
    return collection_name in existing


def load_chroma_store(
    embeddings: Embeddings,
    settings: Settings,
    collection_name: str,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_metadata=COSINE_COLLECTION_METADATA,
    )


def build_chroma_store(
    documents: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    collection_name: str,
) -> Chroma:
    if not documents:
        raise ValueError("Cannot build a vector store without documents.")

    with log_timing("vector_store_creation"):
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=settings.chroma_persist_dir,
            collection_metadata=COSINE_COLLECTION_METADATA,
        )


def get_or_build_chroma_store(
    documents: list[Document],
    chunks: list[Document],
    embeddings: Embeddings,
    settings: Settings,
    fingerprint: str,
) -> tuple[Chroma, bool]:
    collection_name = collection_name_for_fingerprint(fingerprint)
    if collection_exists(settings, collection_name):
        with log_timing("vector_store_reuse"):
            return load_chroma_store(embeddings, settings, collection_name), True

    vectorstore = build_chroma_store(chunks, embeddings, settings, collection_name)
    return vectorstore, False
