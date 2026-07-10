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
    llm_provider: str = "groq",
) -> str:


    if summary_type not in SUMMARY_PROMPTS:
        raise ValueError(f"Unsupported summary type: {summary_type}")

    prompt = SUMMARY_PROMPTS[summary_type]
    chain = prompt | llm | StrOutputParser()

    model_id = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    try:
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

    batch_summaries: list[str] = [""] * len(batches)

    # Prevent Groq 429 rate limits: summaries run in parallel map calls.
    # Use low concurrency to stay within Groq quota.
    max_workers = min(2, max(1, len(batches)))

    def _invoke_one(i: int, batch: str) -> tuple[int, str]:
        from utils.llm_invoke import invoke_llm

        provider = llm_provider
        original_model = (
            getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
        )
        # Only Groq has a remote fallback model; for Gemini/Ollama stay on the same model.
        fallback = (
            "meta-llama/llama-4-scout-17b-16e-instruct"
            if llm_provider == "groq"
            else original_model
        )

        def _set_model(m: str):
            try:
                setattr(llm, "model", m)
            except Exception:
                pass

        summary_text = invoke_llm(
            provider=provider,
            original_model=original_model,
            chain_name="Summary",
            fallback_model=fallback,
            max_retries=3,
            model_setter=_set_model,
            chain_callable=lambda: chain.invoke({"context": batch}),
            inputs={"context": batch},
        )
        return i, summary_text

    with log_timing("summary_generation"):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_invoke_one, i, batch) for i, batch in enumerate(batches)]

            for fut in as_completed(futures):
                i, summary = fut.result()
                batch_summaries[i] = summary

    if len(batch_summaries) == 1:
        return batch_summaries[0]

    combined_context = "\n\n".join(
        f"Partial summary {index}:\n{summary}"
        for index, summary in enumerate(batch_summaries, start=1)
    )

    # Prevent 413 request_too_large on the reduce step as well.
    max_combined_chars = 12000
    if len(combined_context) > max_combined_chars:
        combined_context = combined_context[:max_combined_chars]

    from utils.llm_invoke import invoke_llm

    provider = llm_provider
    original_model = (
        getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
    )
    # Only Groq has a remote fallback model; for Gemini/Ollama stay on the same model.
    fallback = (
        "meta-llama/llama-4-scout-17b-16e-instruct"
        if llm_provider == "groq"
        else original_model
    )

    def _set_model(m: str):
        try:
            setattr(llm, "model", m)
        except Exception:
            pass

    return invoke_llm(
        provider=provider,
        original_model=original_model,
        chain_name="Summary",
        fallback_model=fallback,
        max_retries=3,
        model_setter=_set_model,
        chain_callable=lambda: chain.invoke({"context": combined_context}),
        inputs={"context": combined_context},
    )


