from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import SUMMARY_PROMPTS


def _batch_documents(documents: list[Document], max_chars: int = 12000) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_chars = 0

    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "N/A")
        text = f"[Source: {source} | Page: {page}]\n{document.page_content}"

        if current and current_chars + len(text) > max_chars:
            batches.append("\n\n".join(current))
            current = []
            current_chars = 0

        current.append(text)
        current_chars += len(text)

    if current:
        batches.append("\n\n".join(current))

    return batches


def generate_summary(
    llm: BaseChatModel,
    documents: list[Document],
    summary_type: str,
) -> str:
    """Summarize PDFs with a map-reduce style LCEL flow."""

    if summary_type not in SUMMARY_PROMPTS:
        raise ValueError(f"Unsupported summary type: {summary_type}")

    prompt = SUMMARY_PROMPTS[summary_type]
    chain = prompt | llm | StrOutputParser()

    batch_summaries = [
        chain.invoke({"context": batch}) for batch in _batch_documents(documents)
    ]

    if len(batch_summaries) == 1:
        return batch_summaries[0]

    combined_context = "\n\n".join(
        f"Partial summary {index}:\n{summary}"
        for index, summary in enumerate(batch_summaries, start=1)
    )
    return chain.invoke({"context": combined_context})

