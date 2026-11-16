import pytest

from ops_agent.runner import build_agent
from ops_agent.state import Message
from ops_agent.stub_llm import StubLLM

ASHA = "asha.rao@example.com"


@pytest.fixture
def agent():
    return build_agent(StubLLM())


def msg(body: str, subject: str = "Query", sender: str = ASHA) -> Message:
    return Message(id="t-1", sender=sender, subject=subject, body=body)
