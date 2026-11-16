"""The production orchestrator: a LangGraph StateGraph (ADR-0001).

The approval step uses `interrupt`, so a run pauses with its state checkpointed,
waits for a human (minutes or hours), and resumes with `Command(resume=decision)`.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .nodes import (Nodes, route_after_approval, route_after_classify, route_after_decide,
                    route_after_policy)
from .state import AgentState


def build_graph(nodes: Nodes, checkpointer=None):
    def approve(state: AgentState) -> dict:
        request = nodes.approval_request(state)
        decision = interrupt(request)  # pauses here until a human answers
        return nodes.record_decision(state, request, decision)

    t = nodes.tracer.wrap
    g = StateGraph(AgentState)
    g.add_node("sanitize", t("sanitize", nodes.sanitize))
    g.add_node("classify", t("classify", nodes.classify))
    g.add_node("gather", t("gather", nodes.gather))
    g.add_node("decide", t("decide", nodes.decide))
    g.add_node("draft", t("draft", nodes.draft))
    g.add_node("policy_check", t("policy_check", nodes.policy_check))
    g.add_node("approve", t("approve", approve))
    g.add_node("send", t("send", nodes.send))
    g.add_node("ticket", t("ticket", nodes.ticket))
    g.add_node("escalate", t("escalate", nodes.escalate))

    g.add_edge(START, "sanitize")
    g.add_edge("sanitize", "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"gather": "gather", "escalate": "escalate"})
    g.add_edge("gather", "decide")
    g.add_conditional_edges("decide", route_after_decide,
                            {"draft": "draft", "ticket": "ticket", "escalate": "escalate"})
    g.add_edge("draft", "policy_check")
    g.add_conditional_edges("policy_check", route_after_policy,
                            {"approve": "approve", "escalate": "escalate"})
    g.add_conditional_edges("approve", route_after_approval,
                            {"send": "send", "escalate": "escalate"})
    for terminal in ("send", "ticket", "escalate"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
