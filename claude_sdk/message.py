"""Message construction with role, content, and token estimation."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    if not text:
        return 0
    # Count words + punctuation tokens as a rough heuristic
    return max(1, len(text) // 4)


@dataclass
class ContentBlock:
    """A single content block within a message."""

    type: str  # "text", "image", "tool_use", "tool_result"
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        d.update(self.data)
        return d

    def token_count(self) -> int:
        if self.text:
            return _estimate_tokens(self.text)
        return 0


@dataclass
class Message:
    """A single message in a Claude conversation."""

    role: str  # "user", "assistant", "system"
    content: str | Sequence[ContentBlock] = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    # ---- convenience constructors ----

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=text)

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=text)

    @classmethod
    def from_blocks(cls, role: str, blocks: Sequence[ContentBlock]) -> Message:
        return cls(role=role, content=blocks)

    # ---- helpers ----

    def text_content(self) -> str:
        """Return plain-text representation of content."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
        return "\n".join(parts)

    def content_blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [ContentBlock(type="text", text=self.content)]
        return list(self.content)

    def token_count(self) -> int:
        """Estimate total tokens in this message."""
        total = _estimate_tokens(self.role)
        if isinstance(self.content, str):
            total += _estimate_tokens(self.content)
        else:
            for block in self.content:
                total += block.token_count()
        return total

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to the dict format expected by the Messages API."""
        d: dict[str, Any] = {"role": self.role}
        if isinstance(self.content, str):
            d["content"] = self.content
        else:
            d["content"] = [b.to_dict() for b in self.content]
        return d

    def __repr__(self) -> str:
        txt = self.text_content()
        preview = txt[:60] + "…" if len(txt) > 60 else txt
        return f"Message(role={self.role!r}, text={preview!r}, tokens~={self.token_count()})"
