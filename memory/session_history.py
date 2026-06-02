from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class TrimmedChatMessageHistory(BaseChatMessageHistory):
    """In-memory chat history that keeps only the newest messages."""

    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages = []


class SessionHistoryStore:
    """Small registry used by RunnableWithMessageHistory."""

    def __init__(self, max_messages: int):
        self.max_messages = max_messages
        self._store: dict[str, TrimmedChatMessageHistory] = {}

    def get_history(self, session_id: str) -> TrimmedChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = TrimmedChatMessageHistory(self.max_messages)
        return self._store[session_id]

    def clear(self, session_id: str) -> None:
        self.get_history(session_id).clear()

