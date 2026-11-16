"""Deterministic guardrails that run before and around every LLM call.

These are plain code on purpose: they must hold even when the model is wrong
or has been manipulated by the content it is reading.
"""

from __future__ import annotations

import re

# Internal reference IDs are kept: the agent needs them to look things up,
# and they mean nothing outside the bank's systems.
_REFERENCE = re.compile(r"\b(?:ACC|PAY|TCK)-\d+\b")

# Order matters: card numbers before generic long digit runs.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?:\+\d{1,3}[ -]?)?\b\d{10}\b|\+\d{1,3}[ -]?\d{3,5}[ -]?\d{3,5}[ -]?\d{3,5}")),
    ("ACCOUNT_NO", re.compile(r"\b\d{9,18}\b")),
]


def redact(text: str) -> tuple[str, dict[str, str]]:
    """Replace PII with numbered placeholders. Returns the text and a placeholder map."""
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    protected: dict[str, str] = {}

    def _protect(m: re.Match[str]) -> str:
        token = f"\x00{len(protected)}\x00"
        protected[token] = m.group(0)
        return token

    working = _REFERENCE.sub(_protect, text)

    for label, pattern in _PII_PATTERNS:
        def _sub(m: re.Match[str], label: str = label) -> str:
            value = m.group(0)
            for placeholder, original in mapping.items():
                if original == value:
                    return placeholder
            counters[label] = counters.get(label, 0) + 1
            placeholder = f"<{label}_{counters[label]}>"
            mapping[placeholder] = value
            return placeholder

        working = pattern.sub(_sub, working)

    for token, original in protected.items():
        working = working.replace(token, original)
    return working, mapping


_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instruction|rule|prompt|polic)", re.I)),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend to be|new instructions)\b", re.I)),
    ("system_prompt_probe", re.compile(r"\b(system prompt|developer message|hidden instructions)\b", re.I)),
    ("fake_markup", re.compile(r"<\s*/?\s*(system|assistant|instructions?)\s*>", re.I)),
    ("exfiltration_request", re.compile(
        r"\b(send|forward|email|cc|bcc)\b[^.\n]{0,60}\bto\b[^.\n]{0,40}[\w.+-]+@[\w-]+\.[\w.-]+", re.I)),
]


def screen_injection(text: str) -> list[str]:
    """Return the names of injection patterns found in untrusted text."""
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


_ESCALATION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("fraud", re.compile(
        r"\b(fraud\w*|unauthori[sz]ed|scam\w*|stolen|did ?n[o']?t (make|authori[sz]e))\b", re.I)),
    ("legal", re.compile(r"\b(lawyer|solicitor|attorney|legal action|sue|court|legal notice)\b", re.I)),
    ("complaint", re.compile(r"\b(complain\w*|unacceptable|ombudsman|escalate this)\b", re.I)),
]


def rule_escalations(text: str) -> list[str]:
    """Deterministic must-escalate categories. Independent of the LLM by design."""
    return [name for name, pattern in _ESCALATION_RULES if pattern.search(text)]


_PLACEHOLDER = re.compile(r"<(EMAIL|CARD|PHONE|ACCOUNT_NO)_\d+>")
_FORBIDDEN_CLAIMS = re.compile(
    r"\b(guarantee\w*|we promise|will definitely|refund (?:will|has been) (?:be )?(?:issued|processed)"
    r"|invest(?:ment)? advice|you should (?:buy|sell|invest))\b",
    re.I,
)


def leaked_placeholders(text: str) -> list[str]:
    return _PLACEHOLDER.findall(text)


def forbidden_claims(text: str) -> list[str]:
    return [m.group(0) for m in _FORBIDDEN_CLAIMS.finditer(text)]
