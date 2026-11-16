"""Wiring helpers and a synchronous harness.

`run_message` executes the same nodes and routers as the LangGraph graph, in the same
order, with a callback in place of the interrupt. It has no LangGraph dependency, which
keeps unit tests and offline evals fast. `tests/test_graph_parity.py` checks that it and
the real graph reach the same outcome on every eval case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import Settings
from .llm import LLM
from .nodes import (Nodes, route_after_approval, route_after_classify, route_after_decide,
                    route_after_policy)
from .state import AgentState, Message
from .tools import SyntheticBackend, ToolExecutor
from .tracing import Tracer

Approver = Callable[[dict[str, Any]], dict[str, Any]]


def auto_approve(request: dict[str, Any]) -> dict[str, Any]:
    """Approve exactly what was shown. Used in evals, which measure the agent's decisions."""
    return {"approved": True, "approver": "eval-harness", "payload_hash": request["payload_hash"]}


def reject_all(request: dict[str, Any]) -> dict[str, Any]:
    return {"approved": False, "approver": "test", "payload_hash": request["payload_hash"]}


@dataclass
class Agent:
    nodes: Nodes
    backend: SyntheticBackend
    tracer: Tracer


def build_agent(llm: LLM, settings: Settings | None = None,
                backend: SyntheticBackend | None = None) -> Agent:
    settings = settings or Settings()
    backend = backend or SyntheticBackend()
    tracer = Tracer(settings)
    nodes = Nodes(llm, ToolExecutor(backend.tools()), settings, tracer)
    return Agent(nodes, backend, tracer)


def _apply(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key == "audit":
            state.setdefault("audit", []).extend(value)
        else:
            state[key] = value


def run_message(agent: Agent, message: Message, approver: Approver = auto_approve) -> AgentState:
    n, t = agent.nodes, agent.tracer.wrap
    state: dict = {"message": message, "audit": []}

    def step(name: str, fn: Callable[[dict], dict]) -> None:
        _apply(state, t(name, fn)(state))

    step("sanitize", n.sanitize)
    step("classify", n.classify)
    if route_after_classify(state) == "gather":
        step("gather", n.gather)
        step("decide", n.decide)
        nxt = route_after_decide(state)
        if nxt == "draft":
            step("draft", n.draft)
            step("policy_check", n.policy_check)
            if route_after_policy(state) == "approve":
                def approve(s: dict) -> dict:
                    request = n.approval_request(s)
                    return n.record_decision(s, request, approver(request))
                step("approve", approve)
                if route_after_approval(state) == "send":
                    step("send", n.send)
                    return state  # type: ignore[return-value]
        elif nxt == "ticket":
            step("ticket", n.ticket)
            return state  # type: ignore[return-value]
    step("escalate", n.escalate)
    return state  # type: ignore[return-value]
