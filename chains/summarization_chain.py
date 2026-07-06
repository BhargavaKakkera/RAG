
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import SUMMARY_PROMPTS
from utils.timing import log_timing


def _batch_documents(documents: list[Document], max_chars: int = 4000) -> list[str]:
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
    """
    Summarize PDFs with a map-reduce style flow.

    Optimizations:
    - Build batches once.
    - Parallelize map-step LLM calls across batches.
    - Reduce unnecessary LLM calls by collapsing when only one batch is produced.
    """

    if summary_type not in SUMMARY_PROMPTS:
        raise ValueError(f"Unsupported summary type: {summary_type}")

    prompt = SUMMARY_PROMPTS[summary_type]
    chain = prompt | llm | StrOutputParser()

    # Debug: confirm what model wrapper thinks it will call.
    # Groq is OpenAI-compatible; some logs may show "openapi" wording even when the model is correct.
    model_id = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    try:
        # Avoid crashing if model attr isn't accessible.
        from logging import getLogger

        getLogger("rag.summary").info(
            "Summary generation using llm model attr=%s summary_type=%s",
            model_id,
            summary_type,
        )
    except Exception:
        pass

    with log_timing("summary_batching"):
        batches = _batch_documents(documents)

    if not batches:
        return ""

    # Map step (parallel)
    batch_summaries: list[str] = [""] * len(batches)
    max_workers = min(8, max(1, len(batches)))

    def _invoke_one(i: int, batch: str) -> tuple[int, str]:
        return i, chain.invoke({"context": batch})

    with log_timing("summary_generation"):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_invoke_one, i, batch) for i, batch in enumerate(batches)]
            for fut in as_completed(futures):
                i, summary = fut.result()
                batch_summaries[i] = summary

    # Reduce step
    if len(batch_summaries) == 1:
        return batch_summaries[0]

    combined_context = "\n\n".join(
        f"Partial summary {index}:\n{summary}"
        for index, summary in enumerate(batch_summaries, start=1)
    )
    return chain.invoke({"context": combined_context})

