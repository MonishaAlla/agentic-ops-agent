"""Graph nodes and routing functions.

Each node takes the current state and returns a partial update. Routing functions
only read state. The same nodes and routers are used by the LangGraph graph
(`graph.py`) and by the synchronous harness (`runner.py`), so they cannot drift apart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from . import guardrails, prompts
from .config import Settings
from .llm import LLM, StructuredOutputError, structured
from .state import (MUST_ESCALATE_INTENTS, AgentState, Classification, Draft, Intent, Outcome,
                    PolicyVerdict, ToolPlan, ToolResult)
from .tools import Approval, RiskTier, ToolExecutor, payload_hash
from .tracing import Tracer


def _event(node: str, **detail: Any) -> dict[str, Any]:
    return {"at": datetime.now(timezone.utc).isoformat(), "node": node, **detail}


class Nodes:
    def __init__(self, llm: LLM, executor: ToolExecutor, settings: Settings, tracer: Tracer):
        self.llm, self.executor, self.settings, self.tracer = llm, executor, settings, tracer

    # ------------------------------------------------------------------ sanitise
    def sanitize(self, state: AgentState) -> dict:
        msg = state["message"]
        text = f"{msg.subject}\n{msg.body}"
        redacted, mapping = guardrails.redact(msg.body)
        flags = guardrails.screen_injection(text)
        rules = guardrails.rule_escalations(text)
        return {
            "redacted_body": redacted,
            "redactions": mapping,
            "injection_flags": flags,
            "rule_escalations": rules,
            "escalation_reasons": [],
            "audit": [_event("sanitize", redacted=len(mapping), injection_flags=flags, rules=rules)],
        }

    # ------------------------------------------------------------------ classify
    def classify(self, state: AgentState) -> dict:
        msg = state["message"]
        reasons = list(state.get("escalation_reasons", []))
        classification: Classification | None = None
        try:
            classification, responses = structured(
                self.llm, Classification, role="fast", system=prompts.CLASSIFY_SYSTEM,
                prompt=prompts.email_block(msg.subject, state["redacted_body"]))
            self.tracer.record_llm(responses)
        except StructuredOutputError:
            reasons.append("classifier_failed")

        # Two independent escalation signals (design 4.2): rules OR model, either escalates.
        reasons += [f"rule:{r}" for r in state.get("rule_escalations", [])]
        reasons += [f"injection:{f}" for f in state.get("injection_flags", [])]
        if classification is not None:
            if classification.intent in MUST_ESCALATE_INTENTS:
                reasons.append(f"intent:{classification.intent.value}")
            if classification.confidence < self.settings.confidence_threshold:
                reasons.append("low_confidence")

        return {
            "classification": classification,
            "escalation_reasons": reasons,
            "audit": [_event("classify",
                             result=classification.model_dump(mode="json") if classification else None,
                             escalation_reasons=reasons)],
        }

    # ------------------------------------------------------------------ gather context
    def gather(self, state: AgentState) -> dict:
        msg, cls = state["message"], state["classification"]
        read_tools = self.executor.describe(RiskTier.READ)
        system = prompts.PLAN_SYSTEM.format(tools=json.dumps(read_tools, indent=2))
        prompt = f"Intent: {cls.intent.value}\n\n{prompts.email_block(msg.subject, state['redacted_body'])}"

        results: list[ToolResult] = []
        refused: list[str] = []
        try:
            plan, responses = structured(self.llm, ToolPlan, role="fast", system=system, prompt=prompt)
            self.tracer.record_llm(responses)
        except StructuredOutputError:
            plan = ToolPlan()

        for call in plan.calls:
            # The model may only *read* in this stage, whatever it asked for.
            if self.executor.tier_of(call.name) is not RiskTier.READ:
                refused.append(call.name)
                continue
            results.append(self.executor.run(call.name, call.args, requester=msg.sender))

        return {
            "tool_results": results,
            "audit": [_event("gather",
                             calls=[{"name": r.name, "args": r.args, "ok": r.ok} for r in results],
                             refused=refused)],
        }

    # ------------------------------------------------------------------ decide
    def decide(self, state: AgentState) -> dict:
        cls = state["classification"]
        results = state.get("tool_results", [])
        reasons = list(state.get("escalation_reasons", []))
        ok = [r for r in results if r.ok and _has_content(r)]

        if cls.intent is Intent.DOCUMENT_SUBMISSION:
            action = "ticket"
        elif any(not r.ok for r in results):
            action, reasons = "escalate", reasons + ["tool_failed"]
        elif not ok:
            action, reasons = "escalate", reasons + ["no_context"]
        else:
            action = "reply"
        return {"action": action, "escalation_reasons": reasons,
                "audit": [_event("decide", action=action)]}

    # ------------------------------------------------------------------ draft
    def draft(self, state: AgentState) -> dict:
        msg = state["message"]
        facts = [{"key": r.fact_key, "tool": r.name, "data": r.data}
                 for r in state.get("tool_results", []) if r.ok]
        prompt = (prompts.email_block(msg.subject, state["redacted_body"])
                  + f"\n\n<tool_results>{json.dumps(facts)}</tool_results>")
        reasons = list(state.get("escalation_reasons", []))
        try:
            draft, responses = structured(self.llm, Draft, role="strong",
                                          system=prompts.DRAFT_SYSTEM, prompt=prompt)
            self.tracer.record_llm(responses)
        except StructuredOutputError:
            return {"escalation_reasons": reasons + ["drafter_failed"],
                    "policy": PolicyVerdict(passed=False, reasons=["no draft"]),
                    "audit": [_event("draft", ok=False)]}
        return {"draft": draft, "audit": [_event("draft", ok=True, cited=draft.cited_facts)]}

    # ------------------------------------------------------------------ policy check
    def policy_check(self, state: AgentState) -> dict:
        if "draft" not in state:
            return {"audit": [_event("policy_check", skipped=True)]}
        draft = state["draft"]
        known = {r.fact_key for r in state.get("tool_results", []) if r.ok}
        problems: list[str] = []

        if not draft.cited_facts:
            problems.append("ungrounded: no facts cited")
        unknown = [f for f in draft.cited_facts if f not in known]
        if unknown:
            problems.append(f"ungrounded: cites unknown facts {unknown}")
        if leaked := guardrails.leaked_placeholders(draft.body):
            problems.append(f"pii_placeholder_leak: {leaked}")
        if claims := guardrails.forbidden_claims(draft.body):
            problems.append(f"forbidden_claims: {claims}")
        if len(draft.body) > self.settings.max_reply_chars:
            problems.append("too_long")

        verdict = PolicyVerdict(passed=not problems, reasons=problems)
        reasons = list(state.get("escalation_reasons", []))
        if problems:
            reasons.append("policy_failed")
        return {"policy": verdict, "escalation_reasons": reasons,
                "audit": [_event("policy_check", passed=verdict.passed, reasons=problems)]}

    # ------------------------------------------------------------------ approval
    def approval_request(self, state: AgentState) -> dict[str, Any]:
        """The exact payload a human is asked to approve. Recipient is fixed by code."""
        msg = state["message"]
        args = {"to": msg.sender, "subject": f"Re: {msg.subject}", "body": state["draft"].body}
        return {"tool": "send_reply", "args": args, "payload_hash": payload_hash("send_reply", args)}

    def record_decision(self, state: AgentState, request: dict, decision: dict) -> dict:
        approved = bool(decision.get("approved")) and decision.get("payload_hash") == request["payload_hash"]
        reasons = list(state.get("escalation_reasons", []))
        if not approved:
            reasons.append("human_rejected")
        return {"approval": {**decision, "approved": approved, "request": request},
                "escalation_reasons": reasons,
                "audit": [_event("approve", approved=approved, approver=decision.get("approver"))]}

    # ------------------------------------------------------------------ terminal actions
    def send(self, state: AgentState) -> dict:
        approval = state["approval"]
        request = approval["request"]
        self.executor.run(request["tool"], request["args"],
                          approval=Approval(approver=approval.get("approver", "unknown"),
                                            payload_hash=approval["payload_hash"]))
        return {"outcome": Outcome.SENT, "sent_to": request["args"]["to"],
                "audit": [_event("send", to=request["args"]["to"])]}

    def ticket(self, state: AgentState) -> dict:
        msg, cls = state["message"], state["classification"]
        r = self.executor.run("create_ticket", {
            "queue": "documents", "summary": f"{msg.subject} (from message {msg.id})",
            "priority": cls.priority.value})
        return {"outcome": Outcome.TICKETED, "ticket_id": (r.data or {}).get("ticket_id"),
                "audit": [_event("ticket", ticket=(r.data or {}).get("ticket_id"))]}

    def escalate(self, state: AgentState) -> dict:
        msg = state["message"]
        cls = state.get("classification")
        priority = "urgent" if any(r.startswith(("rule:fraud", "intent:fraud"))
                                   for r in state.get("escalation_reasons", [])) else (
            cls.priority.value if cls else "high")
        reasons = state.get("escalation_reasons", []) or ["unspecified"]
        r = self.executor.run("create_ticket", {
            "queue": "human_review",
            "summary": f"{msg.subject} (message {msg.id}); reasons: {', '.join(reasons)}",
            "priority": priority})
        return {"outcome": Outcome.ESCALATED, "ticket_id": (r.data or {}).get("ticket_id"),
                "audit": [_event("escalate", reasons=reasons, priority=priority)]}


def _has_content(result: ToolResult) -> bool:
    data = result.data or {}
    return not ("matches" in data and not data["matches"])


# ---------------------------------------------------------------------- routers
def route_after_classify(state: AgentState) -> str:
    return "escalate" if state.get("escalation_reasons") else "gather"


def route_after_decide(state: AgentState) -> str:
    return {"reply": "draft", "ticket": "ticket"}.get(state["action"], "escalate")


def route_after_policy(state: AgentState) -> str:
    policy = state.get("policy")
    return "approve" if policy is not None and policy.passed else "escalate"


def route_after_approval(state: AgentState) -> str:
    approval = state.get("approval") or {}
    return "send" if approval.get("approved") else "escalate"
