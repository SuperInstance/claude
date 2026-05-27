"""Cost estimation for Claude API usage."""

from __future__ import annotations

from dataclasses import dataclass, field

# Prices per million tokens (USD) — as of early 2025
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class CostEntry:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Track estimated costs across multiple API calls."""

    entries: list[CostEntry] = field(default_factory=list)
    default_model: str = "claude-3.5-sonnet"

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """Return estimated cost in USD for a single call."""
        model = model or self.default_model
        pricing = _MODEL_PRICING.get(model, _MODEL_PRICING["claude-3.5-sonnet"])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return cost

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> CostEntry:
        """Record a call and return the CostEntry."""
        model = model or self.default_model
        cost = self.estimate(input_tokens, output_tokens, model)
        entry = CostEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.entries.append(entry)
        return entry

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(e.input_tokens for e in self.entries),
            sum(e.output_tokens for e in self.entries),
        )

    def by_model(self) -> dict[str, float]:
        breakdown: dict[str, float] = {}
        for e in self.entries:
            breakdown[e.model] = breakdown.get(e.model, 0.0) + e.cost_usd
        return breakdown

    def __repr__(self) -> str:
        inp, out = self.total_tokens()
        return (
            f"CostTracker(calls={len(self.entries)}, "
            f"tokens={inp}in/{out}out, "
            f"total=${self.total_cost():.4f})"
        )
