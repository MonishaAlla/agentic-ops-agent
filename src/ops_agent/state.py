"""Typed state shared by every node in the graph."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Intent(str, Enum):
    PAYMENT_STATUS = "payment_status"
    ACCOUNT_STATUS = "account_status"
    DOCUMENT_SUBMISSION = "document_submission"
    GENERAL_QUERY = "general_query"
    COMPLAINT = "complaint"
    FRAUD_REPORT = "fraud_report"
    LEGAL = "legal"


# Regulated categories: always handled by a human, whatever the model's confidence.
MUST_ESCALATE_INTENTS = {Intent.COMPLAINT, Intent.FRAUD_REPORT, Intent.LEGAL}


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Outcome(str, Enum):
    SENT = "sent"
    TICKETED = "ticketed"
    ESCALATED = "escalated"


class Message(BaseModel):
    id: str
    sender: str
    subject: str
    body: str


class Classification(BaseModel):
    intent: Intent
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list, max_length=5)


class ToolResult(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None

    @property
    def fact_key(self) -> str:
        """Stable key a draft uses to cite this result, e.g. 'get_payment_status:PAY-5001'."""
        first = next(iter(self.args.values()), "")
        return f"{self.name}:{first}"


class Draft(BaseModel):
    body: str
    cited_facts: list[str] = Field(default_factory=list)


class PolicyVerdict(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    message: Message
    redacted_body: str
    redactions: dict[str, str]
    injection_flags: list[str]
    rule_escalations: list[str]
    classification: Classification | None
    tool_results: list[ToolResult]
    action: Literal["reply", "ticket", "escalate"]
    draft: Draft
    policy: PolicyVerdict
    approval: dict[str, Any] | None
    escalation_reasons: list[str]
    outcome: Outcome
    sent_to: str | None
    ticket_id: str | None
    # Append-only: LangGraph merges each node's events with operator.add.
    audit: Annotated[list[dict[str, Any]], operator.add]
