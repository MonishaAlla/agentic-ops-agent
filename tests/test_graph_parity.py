"""The LangGraph graph and the synchronous harness must reach the same outcome."""

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.types import Command  # noqa: E402

from ops_agent.evals import load_dataset  # noqa: E402
from ops_agent.graph import build_graph  # noqa: E402
from ops_agent.runner import auto_approve, build_agent, run_message  # noqa: E402
from ops_agent.state import Message  # noqa: E402
from ops_agent.stub_llm import StubLLM  # noqa: E402

CASES = load_dataset(Path(__file__).parent.parent / "evals" / "dataset.jsonl")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_graph_matches_harness(case):
    message = Message(id=case.id, sender=case.sender, subject=case.subject, body=case.body)

    expected = run_message(build_agent(StubLLM()), message, auto_approve)["outcome"]

    agent = build_agent(StubLLM())
    graph = build_graph(agent.nodes)
    config = {"configurable": {"thread_id": case.id}}
    result = graph.invoke({"message": message, "audit": []}, config)
    if "__interrupt__" in result:  # paused for human approval
        request = result["__interrupt__"][0].value
        result = graph.invoke(Command(resume=auto_approve(request)), config)

    assert result["outcome"] == expected
