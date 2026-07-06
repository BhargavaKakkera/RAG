
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from memory.session_history import SessionHistoryStore
from prompts.templates import (
    CONDENSE_QUESTION_PROMPT,
    MODE_INSTRUCTIONS,
    NOT_FOUND_RESPONSE,
    QA_PROMPT,
)
from retrievers.threshold_retriever import ThresholdRetriever, adaptive_retrieval_k
from utils.question_rewrite import needs_question_rewrite
from utils.sources import format_documents_for_prompt
from utils.timing import log_pipeline_summary, log_timing

timing_logger = logging.getLogger("rag.timing")


def create_conversational_rag_chain(
    llm: BaseChatModel,
    retriever: ThresholdRetriever,
    history_store: SessionHistoryStore,
    max_context_chars: int | None = None,
    max_retrieval_k: int | None = None,
    max_rewrite_history_messages: int = 6,
    answer_mode: str = "Normal Mode",
    llm_provider: str = "unknown",
    llm_model: str = "unknown",
) -> RunnableWithMessageHistory:
    """Create an LCEL conversational RAG chain with message history."""

    question_rewriter = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    answer_chain = QA_PROMPT | llm | StrOutputParser()
    effective_max_k = max_retrieval_k or retriever.k

    def retrieve_context(inputs: dict[str, Any]) -> dict[str, Any]:
        history = inputs.get("history", [])
        question = inputs["question"]
        stage_timings: dict[str, float] = {}
        total_start = time.perf_counter()

        rewrite_start = time.perf_counter()
        rewrite_required = needs_question_rewrite(question, history)
        recent_history = history[-max_rewrite_history_messages:]
        if rewrite_required:
            with log_timing("question_rewriting", skipped="false"):
                standalone_question = question_rewriter.invoke(
                    {"question": question, "history": recent_history}
                )
        else:
            standalone_question = question
            elapsed = time.perf_counter() - rewrite_start
            timing_logger.info(
                "question_rewriting: %.3fs | skipped=true | reason=standalone_question",
                elapsed,
            )
        stage_timings["question_rewriting"] = time.perf_counter() - rewrite_start

        retrieval_k = adaptive_retrieval_k(standalone_question, effective_max_k)
        retrieval_start = time.perf_counter()
        documents = retriever.retrieve(standalone_question, k=retrieval_k)
        stage_timings["retrieval"] = time.perf_counter() - retrieval_start

        prompt_start = time.perf_counter()
        with log_timing(
            "prompt_construction",
            retrieval_k=retrieval_k,
            chunks=len(documents),
        ):
            context = format_documents_for_prompt(
                documents,
                max_chars=max_context_chars,
            )
        stage_timings["prompt_construction"] = time.perf_counter() - prompt_start

        context_chars = len(context)
        stage_timings["qa_total"] = time.perf_counter() - total_start
        inputs["_stage_timings"] = stage_timings
        inputs["_context_chars"] = context_chars
        inputs["_rewrite_required"] = rewrite_required
        return {
            **inputs,
            "standalone_question": standalone_question,
            "source_documents": documents,
            "context": context,
        }

    def answer_from_context(inputs: dict[str, Any]) -> dict[str, Any]:
        documents = inputs.get("source_documents", [])
        stage_timings = dict(inputs.get("_stage_timings", {}))
        context_chars = inputs.get("_context_chars", len(inputs.get("context", "")))

        if not documents:
            return {
                "answer": NOT_FOUND_RESPONSE,
                "source_documents": [],
                "standalone_question": inputs.get("standalone_question", inputs["question"]),
            }

        mode = inputs.get("mode", answer_mode)
        generation_start = time.perf_counter()
        with log_timing(
            "llm_generation",
            context_chars=context_chars,
            history_messages=0,
        ):
            from utils.llm_invoke import invoke_llm

            # History is folded into standalone_question when rewrite runs.
            # Omitting chat history here avoids duplicating turns in local LLM prompts.
            provider = llm_provider
            # Best-effort model id; used only for logging/retry logic.
            original_model = (
                getattr(llm, "model", None)
                or getattr(llm, "model_name", None)
                or llm_provider
            )



            def _set_model(m: str):
                # Best-effort model switch for the current request.
                # Some wrappers can be frozen; ignore failures.
                try:
                    setattr(llm, "model", m)
                except Exception:
                    pass


            answer = invoke_llm(
                provider=llm_provider,
                original_model=original_model,
                chain_name="QA",
                fallback_model="meta-llama/llama-4-scout-17b-16e-instruct",
                model_setter=_set_model,

                chain_callable=lambda: answer_chain.invoke(

                    {
                        "question": inputs.get(
                            "standalone_question",
                            inputs["question"],
                        ),
                        "history": [],
                        "context": inputs["context"],
                        "mode": mode,
                        "mode_instruction": MODE_INSTRUCTIONS.get(
                            mode,
                            MODE_INSTRUCTIONS["Normal Mode"],
                        ),
                    }
                ),
                inputs={
                    "question": inputs.get(
                        "standalone_question",
                        inputs["question"],
                    ),
                    "history": [],
                    "context": inputs["context"],
                    "mode": mode,
                    "mode_instruction": MODE_INSTRUCTIONS.get(
                        mode,
                        MODE_INSTRUCTIONS["Normal Mode"],
                    ),
                },
            )

        stage_timings["llm_generation"] = time.perf_counter() - generation_start
        stage_timings["qa_total"] = stage_timings.get("qa_total", 0.0) + stage_timings[
            "llm_generation"
        ]
        llm_calls = 2 if inputs.get("_rewrite_required") else 1
        log_pipeline_summary(llm_provider, llm_model, stage_timings)
        timing_logger.info(
            "qa_request_details | provider=%s | model=%s | llm_calls=%s | context_chars=%s",
            llm_provider,
            llm_model,
            llm_calls,
            context_chars,
        )

        return {
            "answer": answer,
            "source_documents": documents,
            "standalone_question": inputs["standalone_question"],
        }

    base_chain = RunnableLambda(retrieve_context) | RunnableLambda(answer_from_context)

    return RunnableWithMessageHistory(
        base_chain,
        history_store.get_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )
