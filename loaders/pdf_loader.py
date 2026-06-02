from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


def load_uploaded_pdfs(uploaded_files: Iterable) -> list[Document]:
    """Extract page-level LangChain Documents from Streamlit uploaded PDFs."""

    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            reader = PdfReader(uploaded_file)
        except Exception as exc:
            raise ValueError(f"Could not read PDF '{uploaded_file.name}': {exc}") from exc

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text = text.strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_index,
                    },
                )
            )

    if not documents:
        raise ValueError("No extractable text was found in the uploaded PDFs.")

    return documents

