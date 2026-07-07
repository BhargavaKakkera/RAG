
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
            "Citation rules:\n"
            "- After factual statements, add a concise citation using the chunk metadata "
            "already present in the context, for example: (Source: file.pdf, Page: 3).\n"
            "- Keep citations short and only where they add traceability.\n\n"
            "Selected response mode: {mode}\n"
            "Mode behavior: {mode_instruction}",
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "Retrieved context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer using only the retrieved context and include concise citations.",
        ),
    ]
)


# Output must be valid JSON.
DOCUMENT_METADATA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract document-level metadata from the supplied PDF text.\n"
            "Return ONLY valid JSON (no markdown, no commentary) with the following schema:\n"
            "{{\n"
            '  "title": string,\n'
            '  "summary": string,\n'
            '  "topics": string[],\n'
            '  "keywords": string[],\n'
            '  "table_of_contents": string,\n'
            '  "page_count": number\n'
            "}}\n\n"
            "Rules:\n"
            "- table_of_contents: create a readable TOC using chapter/topic names you infer from headings/page flow.\n"
            "- topics/keywords: prefer concise, non-duplicated strings.\n"
            "- page_count must equal the provided page_count input value.\n"
            "- Use the language found in the text; if unclear, default to English.",
        ),
        ("human", "Document title candidate: {title}\n\nPage count: {page_count}\n\nPDF content (may include multiple pages):\n{context}"),
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

