"""Tool definitions for Claude's tool-use feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDef:
    """Defines a tool that Claude can invoke.

    Parameters
    ----------
    name : str
        Tool name (must be unique within a request).
    description : str
        Human-readable description shown to the model.
    input_schema : dict
        JSON Schema describing the expected input.
    handler : callable or None
        Optional Python callable to execute the tool locally.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None

    def __post_init__(self):
        if not self.input_schema:
            self.input_schema = {"type": "object", "properties": {}}

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize for the tools array in a Messages API request."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, **kwargs: Any) -> Any:
        """Run the handler if one is registered."""
        if self.handler is None:
            raise RuntimeError(f"Tool {self.name!r} has no handler")
        return self.handler(**kwargs)

    @classmethod
    def function(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable], ToolDef]:
        """Decorator that wraps a Python function as a ToolDef."""

        def decorator(fn: Callable) -> ToolDef:
            schema = parameters or {"type": "object", "properties": {}}
            return cls(
                name=name,
                description=description,
                input_schema=schema,
                handler=fn,
            )

        return decorator

    def __repr__(self) -> str:
        return f"ToolDef(name={self.name!r}, description={self.description[:50]!r})"
