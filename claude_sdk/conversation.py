"""Conversation history management with context-window budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .message import Message


@dataclass
class Conversation:
    """Ordered sequence of messages with context-window management."""

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    max_context_tokens: int = 200_000  # claude-3.5-sonnet default

    # ---- mutation ----

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_user(self, text: str) -> Message:
        m = Message.user(text)
        self.add(m)
        return m

    def add_assistant(self, text: str) -> Message:
        m = Message.assistant(text)
        self.add(m)
        return m

    def add_system(self, text: str) -> Message:
        m = Message.system(text)
        self.system_prompt = text
        self.add(m)
        return m

    def pop(self) -> Message | None:
        return self.messages.pop() if self.messages else None

    def clear(self) -> None:
        self.messages.clear()

    # ---- context windowing ----

    def trim_to_context(self, reserve_tokens: int = 4096) -> None:
        """Drop oldest non-system messages until we fit within the budget.

        Always keeps the system prompt and the most recent messages.
        """
        budget = self.max_context_tokens - reserve_tokens
        while self.total_tokens() > budget and len(self.messages) > 1:
            # find first non-system message to drop
            for i, m in enumerate(self.messages):
                if m.role != "system":
                    self.messages.pop(i)
                    break
            else:
                break

    def truncate_to_last_n(self, n: int) -> None:
        """Keep only the last *n* messages."""
        if len(self.messages) > n:
            # preserve leading system messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            non_system = [m for m in self.messages if m.role != "system"]
            kept = non_system[-n:]
            self.messages = system_msgs + kept

    # ---- queries ----

    def total_tokens(self) -> int:
        total = 0
        if self.system_prompt:
            total += len(self.system_prompt) // 4
        return total + sum(m.token_count() for m in self.messages)

    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def to_api_messages(self) -> list[dict]:
        """Serialize all messages for the Messages API."""
        return [m.to_api_dict() for m in self.messages if m.role != "system"]

    def __repr__(self) -> str:
        return (
            f"Conversation(messages={len(self.messages)}, "
            f"tokens~={self.total_tokens()}, "
            f"budget={self.max_context_tokens})"
        )
