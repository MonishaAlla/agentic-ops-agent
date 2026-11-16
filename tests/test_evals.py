from pathlib import Path

from ops_agent.evals import check_gates, load_dataset, macro_f1, percentile, run_evals
from ops_agent.stub_llm import StubLLM

DATASET = Path(__file__).parent.parent / "evals" / "dataset.jsonl"


def test_macro_f1():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert round(macro_f1(["a", "a", "b"], ["a", "b", "b"]), 3) == 0.667


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) == 0.0


def test_dataset_covers_every_category():
    cases = load_dataset(DATASET)
    assert {c.intent for c in cases} >= {"payment_status", "account_status", "document_submission",
                                          "general_query", "complaint", "fraud_report", "legal"}
    assert sum(c.attack for c in cases) >= 5
    assert sum(c.must_escalate for c in cases) >= 5


def test_offline_eval_passes_safety_gates():
    report = run_evals(StubLLM, load_dataset(DATASET))
    assert check_gates(report["metrics"]) == []
