from __future__ import annotations

import re

from langchain_core.messages import BaseMessage


_ANAPHORA_PATTERNS = [
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\btheir\b",
    r"\bhe\b",
    r"\bshe\b",
    r"\bhis\b",
    r"\bher\b",
    r"\bprevious\b",
    r"\babove\b",
    r"\bearlier\b",
    r"\bsame\b",
    r"\balso\b",
    r"\bmore about\b",
    r"\banother\b",
    r"\bcontinue\b",
    r"\bfollow up\b",
    r"\bas mentioned\b",
    r"\bwhat about\b",
    r"\bhow about\b",
]


def needs_question_rewrite(question: str, history: list[BaseMessage]) -> bool:
    """
    Skip the rewrite LLM call when the question is already self-contained.

    Rewrite only when chat history exists and the question likely depends on it.
    """

    if not history:
        return False

    q = question.strip().lower()
    if not q:
        return False

    if any(re.search(pattern, q) for pattern in _ANAPHORA_PATTERNS):
        return True

    # Very short follow-ups ("why?", "explain more") usually refer to prior turns.
    if len(q.split()) <= 4:
        return True

    return False
