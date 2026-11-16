"""Run one message through the agent and print the trace.

    python -m ops_agent.cli --subject "Payment PAY-5002" --body "Where is my payment PAY-5002?"
"""

from __future__ import annotations

import argparse
import json

from .llm import provider_from_env
from .runner import build_agent, run_message
from .state import Message


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="stub", choices=["stub", "openai", "azure", "bedrock"])
    p.add_argument("--sender", default="customer@example.com")
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True)
    args = p.parse_args()

    agent = build_agent(provider_from_env(args.provider))

    def ask_human(request: dict) -> dict:
        print("\n--- approval needed ---")
        print(json.dumps(request["args"], indent=2))
        ok = input("Approve and send? [y/N] ").strip().lower() == "y"
        return {"approved": ok, "approver": "cli-user", "payload_hash": request["payload_hash"]}

    state = run_message(agent, Message(id="cli-1", sender=args.sender,
                                       subject=args.subject, body=args.body), ask_human)
    print("\noutcome:", state["outcome"].value)
    if state.get("escalation_reasons"):
        print("escalation reasons:", ", ".join(state["escalation_reasons"]))
    print("trace:", json.dumps(agent.tracer.summary(), indent=2))


if __name__ == "__main__":
    main()
