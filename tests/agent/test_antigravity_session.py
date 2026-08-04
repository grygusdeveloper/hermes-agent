from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import pytest

from agent.antigravity_session import (
    AntigravityConversation,
    AntigravityConversationExpired,
    _validate_prompt_size,
)
from agent.copilot_acp_client import CopilotACPClient
from agent.model_metadata import get_model_context_length
from run_agent import AIAgent


def _messages(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


def test_second_turn_uses_server_conversation_and_only_new_input(monkeypatch):
    conversation = AntigravityConversation()
    calls: list[dict] = []

    def fake_execute(prompt_text, **kwargs):
        calls.append({"prompt": prompt_text, **kwargs})
        return "answer", "", f"conversation-{len(calls)}"

    monkeypatch.setattr(conversation, "_execute", fake_execute)
    first = _messages(("system", "rules"), ("user", "first question"))
    conversation.run("FULL FIRST PROMPT WITH TOOLS", messages=first, model="gemini")

    second = first + _messages(
        ("assistant", "answer"),
        ("user", "second question"),
    )
    conversation.run("FULL SECOND PROMPT", messages=second, model="gemini")

    assert calls[0]["conversation_id"] is None
    assert calls[0]["prompt"] == "FULL FIRST PROMPT WITH TOOLS"
    assert calls[1]["conversation_id"] == "conversation-1"
    assert "second question" in calls[1]["prompt"]
    assert "first question" not in calls[1]["prompt"]
    assert "answer" not in calls[1]["prompt"]


def test_prefix_change_starts_fresh_after_reset_or_compression(monkeypatch):
    conversation = AntigravityConversation()
    calls: list[str | None] = []

    def fake_execute(prompt_text, **kwargs):
        calls.append(kwargs["conversation_id"])
        return "answer", "", "cid"

    monkeypatch.setattr(conversation, "_execute", fake_execute)
    conversation.run(
        "first full",
        messages=_messages(("system", "rules"), ("user", "first")),
        model="gemini",
    )
    conversation.run(
        "compressed full",
        messages=_messages(("system", "compressed summary"), ("user", "latest")),
        model="gemini",
    )
    assert calls == [None, None]


def test_conversation_state_is_instance_local(monkeypatch):
    left = AntigravityConversation()
    right = AntigravityConversation()
    left_calls: list[str | None] = []
    right_calls: list[str | None] = []

    monkeypatch.setattr(
        left,
        "_execute",
        lambda prompt_text, **kwargs: (
            left_calls.append(kwargs["conversation_id"]) or "left",
            "",
            "left-cid",
        ),
    )
    monkeypatch.setattr(
        right,
        "_execute",
        lambda prompt_text, **kwargs: (
            right_calls.append(kwargs["conversation_id"]) or "right",
            "",
            "right-cid",
        ),
    )
    identical = _messages(("system", "same"), ("user", "same"))
    left.run("same", messages=identical, model="gemini")
    right.run("same", messages=identical, model="gemini")
    assert left_calls == [None]
    assert right_calls == [None]


def test_utf8_prompt_limit_rejects_without_truncation():
    text = "HEAD:" + ("🙂" * 100_000) + ":TAIL"
    with pytest.raises(RuntimeError, match="no context was sent or truncated"):
        _validate_prompt_size(text)
    assert _validate_prompt_size("small") == "small"


def _spool_stream(path: str, size: int, *, tool_name: str = "view_file") -> str:
    conversation_id = "12345678-1234-1234-1234-123456789abc"
    events = [
        {
            "event": "init",
            "conversation_id": conversation_id,
            "init": {
                "agent": "hermes-antigravity-acp-spool",
                "model": "gemini-3.6-flash-high",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": "DONE",
                "tool_name": tool_name,
                "tool_info": {
                    "name": tool_name,
                    "parameters": {"AbsolutePath": path},
                    "output": f"17 lines, {size} bytes",
                },
            },
        },
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": "SPOOL_OK",
                "conversation_id": conversation_id,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def test_oversized_prompt_uses_private_spool_and_cleans_up(monkeypatch):
    conversation = AntigravityConversation()
    prompt = "HEAD\n" + ("schema-data\n" * 12_000) + "TAIL"
    observed: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        path = Path(str(kwargs["cwd"])) / "canonical-prompt.txt"
        observed.update(
            command=list(command),
            path=path,
            directory=path.parent,
            file_bytes=path.read_bytes(),
            file_mode=path.stat().st_mode & 0o777,
            directory_mode=path.parent.stat().st_mode & 0o777,
            env=dict(kwargs["env"]),
        )
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            communicate=Mock(
                return_value=(_spool_stream(str(path), len(prompt.encode("utf-8"))), "")
            ),
        )

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fake_popen)
    result = conversation._execute(
        prompt,
        conversation_id=None,
        model="gemini-3.6-flash-high",
        effort="high",
        timeout_seconds=2,
        cwd="/untrusted/project",
        env={"HOME": "/root"},
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--agent") + 1] == "hermes-antigravity-acp-spool"
    assert command[command.index("--output-format") + 1] == "stream-json"
    bootstrap = command[command.index("--print") + 1]
    assert prompt not in bootstrap
    assert len(bootstrap.encode("utf-8")) <= 120_000
    assert observed["file_bytes"] == prompt.encode("utf-8")
    assert observed["file_mode"] == 0o600
    assert observed["directory_mode"] == 0o700
    observed_env = observed["env"]
    assert isinstance(observed_env, dict)
    assert observed_env["TMPDIR"] == str(observed["directory"])
    assert observed_env["TMP"] == str(observed["directory"])
    assert observed_env["TEMP"] == str(observed["directory"])
    assert result == ("SPOOL_OK", "", "12345678-1234-1234-1234-123456789abc")
    assert not Path(str(observed["path"])).exists()
    assert not Path(str(observed["directory"])).exists()


def test_spool_trace_rejects_unexpected_tool_and_still_cleans_up(monkeypatch):
    conversation = AntigravityConversation()
    prompt = "x" * 130_000
    observed: dict[str, Path] = {}

    def fake_popen(command, **kwargs):
        path = Path(str(kwargs["cwd"])) / "canonical-prompt.txt"
        observed["path"] = path
        observed["directory"] = path.parent
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            communicate=Mock(
                return_value=(
                    _spool_stream(str(path), len(prompt), tool_name="run_command"),
                    "",
                )
            ),
        )

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fake_popen)
    with pytest.raises(RuntimeError, match="forbidden tool"):
        conversation._execute(
            prompt,
            conversation_id=None,
            model="gemini-3.6-flash-high",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )
    assert not observed["path"].exists()
    assert not observed["directory"].exists()


def test_spool_trace_rejects_unexpected_path(monkeypatch):
    conversation = AntigravityConversation()
    prompt = "x" * 130_000

    def fake_popen(command, **kwargs):
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            communicate=Mock(
                return_value=(
                    _spool_stream("/tmp/not-the-spool", len(prompt)),
                    "",
                )
            ),
        )

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fake_popen)
    with pytest.raises(RuntimeError, match="unexpected path"):
        conversation._execute(
            prompt,
            conversation_id=None,
            model="gemini-3.6-flash-high",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )


def test_spool_limit_fails_closed_before_launch(monkeypatch):
    conversation = AntigravityConversation()
    popen = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", popen)

    with pytest.raises(RuntimeError, match="spool transport limit exceeded"):
        conversation._execute(
            "x" * 1_000_001,
            conversation_id=None,
            model="gemini",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )
    popen.assert_not_called()


def test_spool_cleanup_when_process_launch_fails(monkeypatch):
    conversation = AntigravityConversation()
    observed: dict[str, Path] = {}

    def fail_popen(command, **kwargs):
        path = Path(str(kwargs["cwd"])) / "canonical-prompt.txt"
        observed["path"] = path
        observed["directory"] = path.parent
        assert path.exists()
        raise FileNotFoundError("agy")

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fail_popen)
    with pytest.raises(RuntimeError, match="AGY executable not found"):
        conversation._execute(
            "x" * 130_000,
            conversation_id=None,
            model="gemini",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )
    assert not observed["path"].exists()
    assert not observed["directory"].exists()


def test_oversized_incremental_turn_uses_spool_without_replaying_prefix(monkeypatch):
    conversation = AntigravityConversation()
    calls: list[tuple[str, str | None]] = []

    def fake_execute(prompt_text, **kwargs):
        calls.append((prompt_text, kwargs["conversation_id"]))
        return "OK", "", "12345678-1234-1234-1234-123456789abc"

    monkeypatch.setattr(conversation, "_execute", fake_execute)
    first = _messages(("system", "rules"), ("user", "first"))
    conversation.run("FULL FIRST", messages=first, model="gemini")
    huge = "SECOND_HEAD" + ("x" * 130_000) + "SECOND_TAIL"
    second = first + _messages(("assistant", "OK"), ("user", huge))
    conversation.run("FULL SECOND", messages=second, model="gemini")

    assert calls[1][1] == "12345678-1234-1234-1234-123456789abc"
    assert calls[1][0].startswith("User:\nSECOND_HEAD")
    assert calls[1][0].endswith("SECOND_TAIL")
    assert "FULL FIRST" not in calls[1][0]


def test_inline_prompt_keeps_zero_tool_json_transport(monkeypatch):
    conversation = AntigravityConversation()
    payload = json.dumps(
        {
            "status": "SUCCESS",
            "response": "INLINE_OK",
            "conversation_id": "12345678-1234-1234-1234-123456789abc",
        }
    )
    commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            communicate=Mock(return_value=(payload, "")),
        )

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fake_popen)
    conversation._execute(
        "small prompt",
        conversation_id=None,
        model="gemini",
        effort="high",
        timeout_seconds=2,
        cwd=None,
        env=None,
    )
    command = commands[0]
    assert command[command.index("--agent") + 1] == "hermes-antigravity-acp"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--print") + 1] == "small prompt"


def test_resume_propagates_unrelated_failure_without_fresh_replay(monkeypatch):
    conversation = AntigravityConversation()
    execute = Mock(side_effect=[("FIRST", "", "cid-1"), RuntimeError("quota")])
    monkeypatch.setattr(conversation, "_execute", execute)
    first = _messages(("user", "first"))
    conversation.run("FULL FIRST", messages=first, model="gemini")

    with pytest.raises(RuntimeError, match="quota"):
        conversation.run(
            "FULL SECOND",
            messages=first + _messages(("assistant", "FIRST"), ("user", "second")),
            model="gemini",
        )

    assert execute.call_count == 2
    assert conversation._conversation_id == "cid-1"


def test_expired_conversation_retries_once_as_fresh(monkeypatch):
    conversation = AntigravityConversation()
    execute = Mock(
        side_effect=[
            ("FIRST", "", "cid-1"),
            AntigravityConversationExpired("conversation not found"),
            ("FRESH", "", "cid-2"),
        ]
    )
    monkeypatch.setattr(conversation, "_execute", execute)
    first = _messages(("user", "first"))
    conversation.run("FULL FIRST", messages=first, model="gemini")
    second = first + _messages(("assistant", "FIRST"), ("user", "second"))

    response, _ = conversation.run("FULL SECOND", messages=second, model="gemini")

    assert response == "FRESH"
    assert [call.kwargs["conversation_id"] for call in execute.call_args_list] == [
        None,
        "cid-1",
        None,
    ]
    assert execute.call_args_list[2].args[0] == "FULL SECOND"
    assert conversation._conversation_id == "cid-2"


def test_client_routes_only_antigravity_marker_to_session_mode(monkeypatch):
    messages = _messages(("user", "hello"))

    antigravity = CopilotACPClient(base_url="acp://antigravity")
    agy_run = Mock(return_value=("AGY_OK", ""))
    monkeypatch.setattr(antigravity._antigravity_conversation, "run", agy_run)
    monkeypatch.setattr(
        antigravity,
        "_run_prompt",
        Mock(side_effect=AssertionError("generic ACP must not run")),
    )
    result = antigravity.chat.completions.create(
        model="gemini-3.6-flash-high",
        messages=messages,
    )
    assert result.choices[0].message.content == "AGY_OK"
    assert agy_run.call_args.kwargs["messages"] == messages

    copilot = CopilotACPClient(base_url="acp://copilot")
    generic_run = Mock(return_value=("COPILOT_OK", ""))
    monkeypatch.setattr(copilot, "_run_prompt", generic_run)
    monkeypatch.setattr(
        copilot._antigravity_conversation,
        "run",
        Mock(side_effect=AssertionError("AGY must not run")),
    )
    result = copilot.chat.completions.create(model="copilot-acp", messages=messages)
    assert result.choices[0].message.content == "COPILOT_OK"
    generic_run.assert_called_once()


def test_gemini_spawn_context_window_is_one_mib_tokens():
    assert get_model_context_length(
        "gemini-3.6-flash-high",
        base_url="acp://antigravity",
        provider="copilot-acp",
    ) == 1_048_576


def test_fresh_call_requires_conversation_id(monkeypatch):
    conversation = AntigravityConversation()
    monkeypatch.setattr(conversation, "_execute", Mock(return_value=("OK", "", "")))

    with pytest.raises(RuntimeError, match="did not return a conversation_id"):
        conversation.run(
            "FULL", messages=_messages(("user", "hello")), model="gemini"
        )

    assert conversation._conversation_id is None
    assert conversation._previous_messages == ()


def test_request_local_clients_share_primary_antigravity_conversation(monkeypatch):
    primary = CopilotACPClient(base_url="acp://antigravity")
    request = CopilotACPClient(base_url="acp://antigravity")
    agent = object.__new__(AIAgent)
    agent.provider = "copilot-acp"
    agent._client_kwargs = {"base_url": "acp://antigravity"}
    agent._ensure_primary_openai_client = Mock(return_value=primary)
    agent._openai_client_lock = Mock(return_value=nullcontext())
    agent._create_openai_client = Mock(return_value=request)

    actual = agent._create_request_openai_client(reason="test", api_kwargs={})

    assert actual is request
    assert request._antigravity_conversation is primary._antigravity_conversation


def test_two_request_clients_resume_one_primary_antigravity_conversation():
    primary = CopilotACPClient(base_url="acp://antigravity")
    first_request = CopilotACPClient(base_url="acp://antigravity")
    second_request = CopilotACPClient(base_url="acp://antigravity")
    execute = Mock(side_effect=[("FIRST", "", "cid-1"), ("SECOND", "", "cid-2")])
    primary._antigravity_conversation._execute = execute
    agent = object.__new__(AIAgent)
    agent.provider = "copilot-acp"
    agent._client_kwargs = {"base_url": "acp://antigravity"}
    agent._ensure_primary_openai_client = Mock(return_value=primary)
    agent._openai_client_lock = Mock(return_value=nullcontext())
    agent._create_openai_client = Mock(side_effect=[first_request, second_request])

    first_messages = _messages(("user", "first"))
    first_client = agent._create_request_openai_client(reason="first", api_kwargs={})
    first_client.chat.completions.create(model="gemini", messages=first_messages)
    first_client.close()

    second_messages = _messages(
        ("user", "first"), ("assistant", "FIRST"), ("user", "second")
    )
    second_client = agent._create_request_openai_client(reason="second", api_kwargs={})
    result = second_client.chat.completions.create(model="gemini", messages=second_messages)

    assert result.choices[0].message.content == "SECOND"
    assert execute.call_count == 2
    assert execute.call_args_list[1].kwargs["conversation_id"] == "cid-1"
    assert execute.call_args_list[1].args[0] == "User:\nsecond"


def test_antigravity_abort_terminates_active_process_group(monkeypatch):
    conversation = AntigravityConversation()
    process = SimpleNamespace(pid=4321, poll=Mock(return_value=None))
    conversation._request_active = True
    conversation._active_process = process
    killpg = Mock()
    monkeypatch.setattr("agent.antigravity_session.os.killpg", killpg)

    conversation.abort()

    killpg.assert_called_once_with(4321, 15)


def test_agent_abort_delegates_to_antigravity_conversation():
    client = CopilotACPClient(base_url="acp://antigravity")
    client._antigravity_conversation.abort = Mock()
    agent = object.__new__(AIAgent)
    agent._client_log_context = Mock(return_value="test")

    agent._abort_request_openai_client(client, reason="interrupt")

    client._antigravity_conversation.abort.assert_called_once_with()


def test_shared_request_close_does_not_abort_primary_conversation():
    primary = CopilotACPClient(base_url="acp://antigravity")
    request = CopilotACPClient(base_url="acp://antigravity")
    agent = object.__new__(AIAgent)
    agent.provider = "copilot-acp"
    agent._client_kwargs = {"base_url": "acp://antigravity"}
    agent._ensure_primary_openai_client = Mock(return_value=primary)
    agent._openai_client_lock = Mock(return_value=nullcontext())
    agent._create_openai_client = Mock(return_value=request)
    bound = agent._create_request_openai_client(reason="test", api_kwargs={})
    primary._antigravity_conversation.abort = Mock()

    bound.close()

    primary._antigravity_conversation.abort.assert_not_called()
    assert bound._owns_antigravity_conversation is False


def test_abort_during_spawn_publication_is_not_lost(monkeypatch):
    conversation = AntigravityConversation()
    popen_entered = threading.Event()
    allow_popen_return = threading.Event()
    killed = threading.Event()

    class Process:
        pid = 424242
        returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            assert killed.wait(timeout=2)
            self.returncode = -15
            return "", "terminated"

    process = Process()

    def fake_popen(*args, **kwargs):
        popen_entered.set()
        assert allow_popen_return.wait(timeout=2)
        return process

    def fake_killpg(pid, sig):
        assert pid == process.pid
        killed.set()

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.antigravity_session.os.killpg", fake_killpg)
    outcome: list[BaseException] = []

    def execute():
        try:
            conversation._execute(
                "hello",
                conversation_id=None,
                model="gemini",
                effort="high",
                timeout_seconds=2,
                cwd=None,
                env=None,
            )
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    assert popen_entered.wait(timeout=2)
    aborter = threading.Thread(target=conversation.abort)
    aborter.start()
    time.sleep(0.05)
    assert aborter.is_alive(), "abort must wait until the child is published"
    allow_popen_return.set()
    worker.join(timeout=3)
    aborter.join(timeout=3)

    assert killed.is_set()
    assert not worker.is_alive()
    assert not aborter.is_alive()
    assert outcome and "AGY failed" in str(outcome[0])


@pytest.mark.parametrize(
    "bad_id, error",
    [
        ({"bad": "shape"}, "non-string conversation_id"),
        ("not-a-uuid", "malformed conversation_id"),
    ],
)
def test_execute_rejects_malformed_conversation_id(monkeypatch, bad_id, error):
    conversation = AntigravityConversation()
    payload = json.dumps(
        {"status": "SUCCESS", "response": "ok", "conversation_id": bad_id}
    )
    process = SimpleNamespace(
        pid=12345,
        returncode=0,
        communicate=Mock(return_value=(payload, "")),
    )
    monkeypatch.setattr(
        "agent.antigravity_session.subprocess.Popen", Mock(return_value=process)
    )

    with pytest.raises(RuntimeError, match=error):
        conversation._execute(
            "hello",
            conversation_id=None,
            model="gemini",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )


def test_late_abort_fails_current_request_without_poisoning_next(monkeypatch):
    conversation = AntigravityConversation()
    valid_id = "12345678-1234-1234-1234-123456789abc"
    payload = json.dumps(
        {"status": "SUCCESS", "response": "ok", "conversation_id": valid_id}
    )

    def make_process(*args, **kwargs):
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            poll=Mock(return_value=0),
            communicate=Mock(return_value=(payload, "")),
        )

    monkeypatch.setattr("agent.antigravity_session.subprocess.Popen", make_process)
    real_loads = json.loads
    parse_count = 0

    def abort_during_first_parse(value):
        nonlocal parse_count
        parse_count += 1
        if parse_count == 1:
            conversation.abort()
        return real_loads(value)

    monkeypatch.setattr("agent.antigravity_session.json.loads", abort_during_first_parse)

    with pytest.raises(RuntimeError, match="AGY request aborted"):
        conversation._execute(
            "first",
            conversation_id=None,
            model="gemini",
            effort="high",
            timeout_seconds=2,
            cwd=None,
            env=None,
        )

    result = conversation._execute(
        "second",
        conversation_id=None,
        model="gemini",
        effort="high",
        timeout_seconds=2,
        cwd=None,
        env=None,
    )
    assert result == ("ok", "", valid_id)
    assert conversation._abort_requested is False
    assert conversation._request_active is False


def test_idle_abort_does_not_poison_next_request(monkeypatch):
    conversation = AntigravityConversation()
    conversation.abort()
    execute_active = Mock(
        return_value=("ok", "", "12345678-1234-1234-1234-123456789abc")
    )
    monkeypatch.setattr(conversation, "_execute_active", execute_active)

    result = conversation._execute(
        "next",
        conversation_id=None,
        model="gemini",
        effort="high",
        timeout_seconds=2,
        cwd=None,
        env=None,
    )

    assert result[0] == "ok"
    execute_active.assert_called_once()


def test_abort_before_atomic_success_transition_cannot_be_lost(monkeypatch):
    conversation = AntigravityConversation()
    before_success_lock = threading.Event()
    allow_success_lock = threading.Event()

    class GateLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.worker_id = None
            self.worker_acquires = 0

        def __enter__(self):
            if threading.get_ident() == self.worker_id:
                self.worker_acquires += 1
                # _execute's first acquisition marks the request active; its
                # second is the atomic success/deactivation transition.
                if self.worker_acquires == 2:
                    before_success_lock.set()
                    assert allow_success_lock.wait(timeout=2)
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._lock.release()

    gate = GateLock()
    conversation._process_lock = gate
    monkeypatch.setattr(
        conversation,
        "_execute_active",
        Mock(return_value=("ok", "", "12345678-1234-1234-1234-123456789abc")),
    )
    outcome: list[object] = []

    def execute():
        gate.worker_id = threading.get_ident()
        try:
            outcome.append(
                conversation._execute(
                    "hello",
                    conversation_id=None,
                    model="gemini",
                    effort="high",
                    timeout_seconds=2,
                    cwd=None,
                    env=None,
                )
            )
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    assert before_success_lock.wait(timeout=2)
    conversation.abort()
    allow_success_lock.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], RuntimeError)
    assert "AGY request aborted" in str(outcome[0])
    assert conversation._request_active is False
    assert conversation._abort_requested is False
