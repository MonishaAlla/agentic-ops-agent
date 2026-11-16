"""Per-run trace: which node ran, how long it took, tokens and cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings
from .llm import LLMResponse


@dataclass
class Span:
    node: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models: list[str] = field(default_factory=list)


class Tracer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.spans: list[Span] = []
        self._current: Span | None = None

    def wrap(self, name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        def traced(state: dict) -> dict:
            span = Span(node=name, duration_ms=0.0)
            self._current = span
            start = time.perf_counter()
            try:
                return fn(state)
            finally:
                span.duration_ms = (time.perf_counter() - start) * 1000
                self.spans.append(span)
                self._current = None
        traced.__name__ = name
        return traced

    def record_llm(self, responses: list[LLMResponse]) -> None:
        if self._current is None:
            return
        for r in responses:
            self._current.input_tokens += r.input_tokens
            self._current.output_tokens += r.output_tokens
            self._current.cost_usd += self.settings.cost(r.model, r.input_tokens, r.output_tokens)
            self._current.models.append(r.model)

    def summary(self) -> dict[str, Any]:
        return {
            "latency_ms": sum(s.duration_ms for s in self.spans),
            "input_tokens": sum(s.input_tokens for s in self.spans),
            "output_tokens": sum(s.output_tokens for s in self.spans),
            "cost_usd": sum(s.cost_usd for s in self.spans),
            "path": [s.node for s in self.spans],
        }
