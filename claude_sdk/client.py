"""ClaudeClient — constructs API requests without making real HTTP calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .conversation import Conversation
from .message import ContentBlock, Message
from .tool import ToolDef
from .cost import CostTracker


@dataclass
class ClaudeClient:
    """Builds request payloads for the Claude Messages API.

    This client does *not* perform HTTP requests — it constructs the
    request body and tracks costs so you can inspect before sending.
    """

    api_key: str = ""
    model: str = "claude-3.5-sonnet"
    max_tokens: int = 4096
    temperature: float = 1.0
    system: str | None = None
    tools: list[ToolDef] = field(default_factory=list)
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    # ---- request building ----

    def build_request(
        self,
        conversation: Conversation,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Construct a Messages API request body."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": conversation.to_api_messages(),
        }

        if temperature is not None:
            body["temperature"] = temperature
        elif self.temperature != 1.0:
            body["temperature"] = self.temperature

        system_text = self.system or conversation.system_prompt
        if system_text:
            body["system"] = system_text

        if self.tools:
            body["tools"] = [t.to_api_dict() for t in self.tools]

        return body

    def simulate_response(
        self,
        assistant_text: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> Message:
        """Create a synthetic assistant Message and record costs."""
        msg = Message.assistant(assistant_text)
        msg.model = self.model

        # estimate tokens if not provided
        if input_tokens is None:
            input_tokens = sum(m.token_count() for m in msg.content_blocks())
        if output_tokens is None:
            output_tokens = msg.token_count()

        msg.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.cost_tracker.record(input_tokens, output_tokens, self.model)
        return msg

    # ---- tool helpers ----

    def add_tool(self, tool: ToolDef) -> None:
        self.tools.append(tool)

    def find_tool(self, name: str) -> ToolDef | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def __repr__(self) -> str:
        return (
            f"ClaudeClient(model={self.model!r}, "
            f"tools={len(self.tools)}, "
            f"cost={self.cost_tracker.total_cost():.4f})"
        )
