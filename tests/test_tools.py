import pytest

from ops_agent.tools import Approval, ApprovalError, SyntheticBackend, ToolExecutor, payload_hash

ASHA, VIKRAM = "asha.rao@example.com", "vikram.s@example.com"


@pytest.fixture
def backend():
    return SyntheticBackend()


@pytest.fixture
def executor(backend):
    return ToolExecutor(backend.tools())


def test_read_tool_returns_data_for_owner(executor):
    r = executor.run("get_payment_status", {"payment_id": "PAY-5001"}, requester=ASHA)
    assert r.ok and r.data["status"] == "completed" and "owner" not in r.data


def test_read_tool_refuses_other_customers_data(executor):
    r = executor.run("get_payment_status", {"payment_id": "PAY-5003"}, requester=ASHA)
    assert not r.ok and r.error.startswith("forbidden")


def test_scoped_tool_without_requester_is_refused(executor):
    assert not executor.run("get_account_status", {"account_id": "ACC-1001"}).ok


def test_unknown_reference_and_unknown_tool(executor):
    assert executor.run("get_payment_status", {"payment_id": "PAY-0"}, requester=ASHA).error.startswith("not found")
    assert executor.run("delete_everything", {}).error == "unknown tool"


def test_bad_arguments_do_not_crash(executor):
    r = executor.run("get_payment_status", {"wrong": "x"}, requester=ASHA)
    assert not r.ok and "bad arguments" in r.error


ARGS = {"to": ASHA, "subject": "Re: hi", "body": "hello"}


def test_external_tool_requires_approval(executor, backend):
    with pytest.raises(ApprovalError):
        executor.run("send_reply", ARGS)
    assert backend.outbox == []


def test_approval_is_bound_to_the_exact_payload(executor, backend):
    approval = Approval("reviewer", payload_hash("send_reply", ARGS))
    with pytest.raises(ApprovalError):
        executor.run("send_reply", {**ARGS, "to": "attacker@evil.com"}, approval=approval)
    assert executor.run("send_reply", ARGS, approval=approval).ok
    assert backend.outbox == [ARGS]
