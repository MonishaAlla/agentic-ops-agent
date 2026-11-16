import pytest

from ops_agent.llm import LLMResponse, StructuredOutputError, structured
from ops_agent.state import Classification


class Scripted:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), 0

    def complete(self, *, role, system, prompt):
        self.calls += 1
        return LLMResponse(self.outputs.pop(0), "scripted", 10, 10, 1.0)


GOOD = '{"intent": "payment_status", "priority": "normal", "confidence": 0.9}'


def test_parses_fenced_json():
    result, _ = structured(Scripted(f"```json\n{GOOD}\n```"), Classification, role="fast", system="", prompt="")
    assert result.intent.value == "payment_status"


def test_retries_once_on_invalid_output():
    llm = Scripted('{"intent": "nonsense"}', GOOD)
    result, responses = structured(llm, Classification, role="fast", system="", prompt="")
    assert llm.calls == 2 and len(responses) == 2 and result.confidence == 0.9


def test_raises_after_retries_exhausted():
    with pytest.raises(StructuredOutputError):
        structured(Scripted("not json", "still not json"), Classification, role="fast", system="", prompt="")
