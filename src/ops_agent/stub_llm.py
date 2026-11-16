"""A deterministic stand-in for an LLM.

It lets the whole pipeline, the tests and the CI eval run offline with no API key.
It is a keyword heuristic, so its eval scores measure the *harness*, not model quality;
real numbers come from running the evals against a real provider.
"""

from __future__ import annotations

import json
import re

from .llm import LLMResponse

_EMAIL = re.compile(r"<email>(.*?)</email>", re.S)
_RESULTS = re.compile(r"<tool_results>(.*?)</tool_results>", re.S)

_KEYWORDS: list[tuple[str, str, str, float]] = [
    # (regex, intent, priority, confidence)
    (r"fraud|unauthori[sz]ed|scam|did ?n[o']?t make", "fraud_report", "urgent", 0.95),
    (r"lawyer|legal action|court|solicitor|legal notice", "legal", "high", 0.93),
    (r"complain|unacceptable|ombudsman|disappointed", "complaint", "high", 0.9),
    (r"cut-?off|opening hours|support hours|statements?\b", "general_query", "low", 0.85),
    (r"attached|kyc|submit\w* (?:the|my) document|documents? (?:attached|enclosed)",
     "document_submission", "low", 0.86),
    (r"\bPAY-\d+|payment|transfer", "payment_status", "normal", 0.9),
    (r"\bACC-\d+|account (?:status|frozen|blocked)|frozen|blocked", "account_status", "normal", 0.88),
]


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class StubLLM:
    def complete(self, *, role: str, system: str, prompt: str) -> LLMResponse:
        if "TASK: CLASSIFY" in system:
            text = self._classify(prompt)
        elif "TASK: PLAN" in system:
            text = self._plan(prompt)
        elif "TASK: DRAFT" in system:
            text = self._draft(prompt)
        else:
            text = "{}"
        return LLMResponse(text, f"stub-{role}", _tokens(system + prompt), _tokens(text), 1.0)

    @staticmethod
    def _email(prompt: str) -> str:
        m = _EMAIL.search(prompt)
        return m.group(1) if m else prompt

    def _classify(self, prompt: str) -> str:
        email = self._email(prompt)
        for pattern, intent, priority, conf in _KEYWORDS:
            if re.search(pattern, email, re.I):
                return json.dumps({"intent": intent, "priority": priority,
                                   "confidence": conf, "reasoning": f"matched {pattern!r}"})
        return json.dumps({"intent": "general_query", "priority": "low",
                           "confidence": 0.5, "reasoning": "no clear signal"})

    def _plan(self, prompt: str) -> str:
        email = self._email(prompt)
        intent = re.search(r"Intent: (\w+)", prompt)
        intent = intent.group(1) if intent else ""
        calls = []
        if intent == "payment_status":
            calls = [{"name": "get_payment_status", "args": {"payment_id": p}}
                     for p in dict.fromkeys(re.findall(r"PAY-\d+", email))]
        elif intent == "account_status":
            calls = [{"name": "get_account_status", "args": {"account_id": a}}
                     for a in dict.fromkeys(re.findall(r"ACC-\d+", email))]
        elif intent == "general_query":
            calls = [{"name": "search_kb", "args": {"query": email.strip()[:200]}}]
        return json.dumps({"calls": calls[:5]})

    def _draft(self, prompt: str) -> str:
        m = _RESULTS.search(prompt)
        results = json.loads(m.group(1)) if m else []
        lines, cited = [], []
        for r in results:
            data = r["data"]
            if "payment_id" in data:
                line = f"Payment {data['payment_id']} for {data['amount']} is {data['status']}"
                line += f" ({data['reason']})." if data.get("reason") else "."
            elif "account_id" in data:
                line = f"Account {data['account_id']} is currently {data['status']}"
                line += f" ({data['reason']})." if data.get("reason") else "."
            elif data.get("matches"):
                line = data["matches"][0]["text"]
            else:
                continue
            lines.append(line)
            cited.append(r["key"])
        body = "Dear customer,\n\nThank you for your message. " + " ".join(lines) + \
               "\n\nIf you need anything further, please reply to this email.\n\nOperations Team"
        return json.dumps({"body": body, "cited_facts": cited})
