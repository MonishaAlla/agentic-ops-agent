"""Run the eval suite and enforce the safety gates.

    python evals/run_evals.py                       # offline, deterministic stub
    python evals/run_evals.py --provider bedrock    # real model (see .env.example)

Exits non-zero if any safety gate fails, so CI blocks the change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ops_agent.evals import GATES, check_gates, load_dataset, run_evals
from ops_agent.llm import provider_from_env

HERE = Path(__file__).parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="stub", choices=["stub", "openai", "azure", "bedrock"])
    p.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    p.add_argument("--out", default="runs/eval_report.json")
    args = p.parse_args()

    cases = load_dataset(args.dataset)
    report = run_evals(lambda: provider_from_env(args.provider), cases)
    metrics = report["metrics"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, **report}, indent=2, default=str))

    print(f"\nprovider: {args.provider}   cases: {metrics['cases']}\n")
    print("| metric | value | gate |\n|---|---|---|")
    for name, value in metrics.items():
        if name == "cases":
            continue
        gate = " ".join(map(str, GATES[name])) if name in GATES else ""
        print(f"| {name} | {value} | {gate} |")

    misses = [r for r in report["rows"] if r["outcome"] != r["expected_outcome"]]
    if misses:
        print("\nOutcome mismatches:")
        for r in misses:
            print(f"  {r['id']}: expected {r['expected_outcome']}, got {r['outcome']} "
                  f"({', '.join(r['escalation_reasons']) or 'no reasons'})")

    failures = check_gates(metrics)
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll safety gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
