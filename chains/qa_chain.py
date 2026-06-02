from __future__ import annotations

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
from retrievers.threshold_retriever import ThresholdRetriever
from utils.sources import format_documents_for_prompt


def create_conversational_rag_chain(
    llm: BaseChatModel,
    retriever: ThresholdRetriever,
    history_store: SessionHistoryStore,
) -> RunnableWithMessageHistory:
    """Create an LCEL conversational RAG chain with message history."""

    question_rewriter = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    answer_chain = QA_PROMPT | llm | StrOutputParser()

    def retrieve_context(inputs: dict[str, Any]) -> dict[str, Any]:
        history = inputs.get("history", [])
        question = inputs["question"]

        if history:
            standalone_question = question_rewriter.invoke(
                {"question": question, "history": history}
            )
        else:
            standalone_question = question

        documents = retriever.retrieve(standalone_question)
        return {
            **inputs,
            "standalone_question": standalone_question,
            "source_documents": documents,
            "context": format_documents_for_prompt(documents),
        }

    def answer_from_context(inputs: dict[str, Any]) -> dict[str, Any]:
        documents = inputs.get("source_documents", [])
        if not documents:
            return {
                "answer": NOT_FOUND_RESPONSE,
                "source_documents": [],
                "standalone_question": inputs.get("standalone_question", inputs["question"]),
            }

        mode = inputs.get("mode", "Normal Mode")
        answer = answer_chain.invoke(
            {
                "question": inputs["question"],
                "history": inputs.get("history", []),
                "context": inputs["context"],
                "mode": mode,
                "mode_instruction": MODE_INSTRUCTIONS.get(
                    mode,
                    MODE_INSTRUCTIONS["Normal Mode"],
                ),
            }
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

