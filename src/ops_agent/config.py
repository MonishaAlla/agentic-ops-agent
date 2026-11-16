"""Runtime settings. Nodes name a model *role*; providers map roles to models (ADR-0004)."""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Fill in from your provider's current price list;
# the eval report uses these to compute cost per message.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "stub-fast": (0.0, 0.0),
    "stub-strong": (0.0, 0.0),
}


@dataclass
class Settings:
    confidence_threshold: float = 0.75
    max_reply_chars: int = 1200
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self.prices.get(model, (0.0, 0.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
