"""Risk-tiered tools and the executor that enforces the tiers (ADR-0002).

The executor, not the prompt, decides what may run. An EXTERNAL tool only runs
with an approval whose hash matches the exact payload being executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .state import ToolResult


class RiskTier(str, Enum):
    READ = "read"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL = "external"


class ApprovalError(PermissionError):
    """Raised when an EXTERNAL tool is called without a matching human approval."""


@dataclass(frozen=True)
class Tool:
    name: str
    tier: RiskTier
    fn: Callable[..., dict[str, Any]]
    description: str
    # Scoped tools receive the requester's identity from the executor, never from the model,
    # and must only return data that requester is entitled to see.
    scoped: bool = False


@dataclass(frozen=True)
class Approval:
    approver: str
    payload_hash: str


def payload_hash(name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ToolExecutor:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def describe(self, tier: RiskTier) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description}
                for t in self._tools.values() if t.tier == tier]

    def tier_of(self, name: str) -> RiskTier | None:
        tool = self._tools.get(name)
        return tool.tier if tool else None

    def run(self, name: str, args: dict[str, Any], *, approval: Approval | None = None,
            requester: str | None = None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, args=args, ok=False, error="unknown tool")

        if tool.tier is RiskTier.EXTERNAL:
            if approval is None:
                raise ApprovalError(f"{name} requires human approval")
            if approval.payload_hash != payload_hash(name, args):
                raise ApprovalError(f"approval does not match the payload for {name}")

        call_args = dict(args)
        if tool.scoped:
            if requester is None:
                return ToolResult(name=name, args=args, ok=False, error="forbidden: no requester")
            call_args["requester"] = requester
        try:
            data = tool.fn(**call_args)
        except TypeError as exc:  # wrong or missing arguments from the planner
            return ToolResult(name=name, args=args, ok=False, error=f"bad arguments: {exc}")
        except PermissionError as exc:
            return ToolResult(name=name, args=args, ok=False, error=f"forbidden: {exc}")
        except LookupError as exc:
            return ToolResult(name=name, args=args, ok=False, error=f"not found: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool failure must never crash the run
            return ToolResult(name=name, args=args, ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(name=name, args=args, ok=True, data=data)


# --------------------------------------------------------------------------- #
# Synthetic backend. Stands in for the core-banking, payments, ticketing and
# mail systems. All data is invented.
# --------------------------------------------------------------------------- #

@dataclass
class SyntheticBackend:
    accounts: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "ACC-1001": {"owner": "asha.rao@example.com", "status": "active", "type": "current"},
        "ACC-1002": {"owner": "vikram.s@example.com", "status": "frozen", "type": "savings",
                     "reason": "KYC documents pending review"},
        "ACC-1003": {"owner": "neha.k@example.com", "status": "closed", "type": "current"},
    })
    payments: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "PAY-5001": {"owner": "asha.rao@example.com", "status": "completed", "amount": "INR 25,000", "date": "2026-11-10"},
        "PAY-5002": {"owner": "asha.rao@example.com", "status": "pending", "amount": "INR 1,20,000", "date": "2026-11-14",
                     "reason": "awaiting beneficiary bank confirmation"},
        "PAY-5003": {"owner": "vikram.s@example.com", "status": "returned", "amount": "INR 8,500", "date": "2026-11-12",
                     "reason": "beneficiary account number invalid"},
    })
    kb: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "KB-01": {"title": "Payment cut-off times",
                  "keywords": "cut-off cutoff cut off same day neft rtgs timing",
                  "text": "Same-day RTGS payments must be submitted before 4:30 pm IST on working days."},
        "KB-02": {"title": "Branch and support hours",
                  "keywords": "hours open timing support branch",
                  "text": "Operations support is available 9 am to 6 pm IST, Monday to Saturday."},
        "KB-03": {"title": "Requesting statements",
                  "keywords": "statement statements download history",
                  "text": "Account statements for the last 24 months can be downloaded from net banking."},
    })
    tickets: list[dict[str, Any]] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _owned(table: dict[str, dict[str, Any]], key: str, requester: str) -> dict[str, Any]:
        if key not in table:
            raise LookupError(key)
        record = table[key]
        if record["owner"].lower() != requester.lower():
            raise PermissionError(f"{key} does not belong to the requester")
        return {k: v for k, v in record.items() if k != "owner"}

    def get_account_status(self, account_id: str, requester: str) -> dict[str, Any]:
        return {"account_id": account_id, **self._owned(self.accounts, account_id, requester)}

    def get_payment_status(self, payment_id: str, requester: str) -> dict[str, Any]:
        return {"payment_id": payment_id, **self._owned(self.payments, payment_id, requester)}

    def search_kb(self, query: str) -> dict[str, Any]:
        words = set(re.findall(r"[a-z]+", query.lower()))
        hits = []
        for article_id, article in self.kb.items():
            score = len(words & set(article["keywords"].split()))
            if score:
                hits.append((score, article_id, article))
        hits.sort(key=lambda h: h[0], reverse=True)
        return {"matches": [{"article_id": a_id, "title": a["title"], "text": a["text"]}
                            for _, a_id, a in hits[:2]]}

    def create_ticket(self, queue: str, summary: str, priority: str = "normal") -> dict[str, Any]:
        ticket_id = f"TCK-{9000 + len(self.tickets) + 1}"
        self.tickets.append({"ticket_id": ticket_id, "queue": queue,
                             "summary": summary, "priority": priority})
        return {"ticket_id": ticket_id}

    def send_reply(self, to: str, subject: str, body: str) -> dict[str, Any]:
        self.outbox.append({"to": to, "subject": subject, "body": body})
        return {"delivered": True}

    def tools(self) -> list[Tool]:
        return [
            Tool("get_account_status", RiskTier.READ, self.get_account_status,
                 "Status of an account. Args: account_id (e.g. ACC-1001).", scoped=True),
            Tool("get_payment_status", RiskTier.READ, self.get_payment_status,
                 "Status of a payment. Args: payment_id (e.g. PAY-5001).", scoped=True),
            Tool("search_kb", RiskTier.READ, self.search_kb,
                 "Search the operations knowledge base. Args: query."),
            Tool("create_ticket", RiskTier.INTERNAL_WRITE, self.create_ticket,
                 "Create an internal ticket. Args: queue, summary, priority."),
            Tool("send_reply", RiskTier.EXTERNAL, self.send_reply,
                 "Send an email reply. Args: to, subject, body."),
        ]
