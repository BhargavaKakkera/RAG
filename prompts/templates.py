from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


NOT_FOUND_RESPONSE = "The answer was not found in the uploaded documents."


MODE_INSTRUCTIONS = {
    "Normal Mode": "Answer clearly and naturally for a general reader.",
    "Beginner Mode": (
        "Explain in simple language, define important terms, and use a small example "
        "when it helps."
    ),
    "Interview Mode": (
        "Answer like a placement interview candidate: start with the core idea, explain "
        "the reasoning, mention practical tradeoffs, and keep the tone confident."
    ),
    "Concise Mode": "Answer in the fewest useful sentences. Avoid extra explanation.",
    "Detailed Expert Mode": (
        "Give a technically rich answer with precise terminology, caveats, and deeper "
        "connections, while still staying grounded in the retrieved context."
    ),
}


CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest user question as a standalone question using the chat "
            "history. Do not answer the question. If it is already standalone, return it "
            "unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Conversational PDF RAG Assistant. Answer only from the retrieved "
            "context. If the context does not contain the answer, respond exactly with: "
            f"{NOT_FOUND_RESPONSE}\n\n"
            "Selected response mode: {mode}\n"
            "Mode behavior: {mode_instruction}",
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "Retrieved context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer using only the retrieved context.",
        ),
    ]
)


SUMMARY_PROMPTS = {
    "Complete Document Summary": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create a complete, well-structured summary of the supplied PDF content. "
                "Preserve important concepts, definitions, examples, and conclusions.",
            ),
            ("human", "PDF content:\n{context}"),
        ]
    ),
    "Chapter/Topic Summaries": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create chapter or topic-level summaries from the supplied PDF content. "
                "Infer topic boundaries from headings, repeated themes, and page flow.",
            ),
            ("human", "PDF content:\n{context}"),
        ]
    ),
    "Key Points": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the most important key points from the supplied PDF content. "
                "Use concise bullets and include concrete facts where present.",
            ),
            ("human", "PDF content:\n{context}"),
        ]
    ),
}

