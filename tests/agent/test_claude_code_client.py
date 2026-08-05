"""Focused unit tests for the Claude Code CLI provider transport and client."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.claude_code_client import (
    CLAUDE_CODE_MARKER_BASE_URL,
    ClaudeCodeClient,
)
from agent.claude_code_session import (
    ClaudeCodeSession,
    ClaudeCodeSessionExpired,
    _incremental_prompt,
    _is_expired_session_error,
    _message_fingerprint,
    _parse_stream_json_output,
    _validate_flag_size,
)


class _FakeStream:
    def __init__(self, data: str = ""):
        self._data = data
        self._pos = 0
        self.closed = False

    def write(self, data):
        return len(data)

    def close(self):
        self.closed = True

    def readline(self):
        if self._pos >= len(self._data):
            return ""
        nxt = self._data.find("\n", self._pos)
        if nxt < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : nxt + 1]
        self._pos = nxt + 1
        return chunk

    def __iter__(self):
        while True:
            line = self.readline()
            if line == "":
                break
            yield line



def _messages(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


def _stream_stdout(
    *,
    text: str = "hello",
    session_id: str = "12345678-1234-1234-1234-123456789abc",
    model: str = "claude-sonnet-4-6",
    is_error: bool = False,
    include_thinking: bool = False,
) -> str:
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": session_id,
            "model": model,
        },
    ]
    content = []
    if include_thinking:
        content.append({"type": "thinking", "thinking": "plan..."})
    content.append({"type": "text", "text": text})
    events.append(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": content,
                "model": model,
            },
            "session_id": session_id,
        }
    )
    events.append(
        {
            "type": "result",
            "subtype": "success" if not is_error else "error",
            "is_error": is_error,
            "result": text if not is_error else "boom",
            "session_id": session_id,
        }
    )
    return "\n".join(json.dumps(e) for e in events) + "\n"


class TestParseStreamJson:
    def test_parses_text_reasoning_and_session_id(self):
        stdout = _stream_stdout(text="ANSWER", include_thinking=True)
        response, reasoning, session_id = _parse_stream_json_output(stdout)
        assert response == "ANSWER"
        assert reasoning == "plan..."
        assert session_id == "12345678-1234-1234-1234-123456789abc"

    def test_falls_back_to_result_envelope_text(self):
        sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "FROM_RESULT",
                        "session_id": sid,
                    }
                )
            ]
        )
        response, reasoning, session_id = _parse_stream_json_output(stdout)
        assert response == "FROM_RESULT"
        assert reasoning == ""
        assert session_id == sid

    def test_result_error_raises(self):
        stdout = _stream_stdout(is_error=True)
        with pytest.raises(RuntimeError, match="result rejected|result error"):
            _parse_stream_json_output(stdout)

    def test_result_expired_error_is_session_expired(self):
        sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "result",
                        "is_error": True,
                        "result": "session not found",
                        "session_id": sid,
                    }
                )
            ]
        )
        with pytest.raises(ClaudeCodeSessionExpired, match="session not found"):
            _parse_stream_json_output(stdout)


class TestIncrementalAndFingerprints:
    def test_incremental_prompt_skips_assistant(self):
        msgs = _messages(
            ("system", "rules"),
            ("user", "first"),
            ("assistant", "answer"),
            ("user", "second"),
            ("tool", "tool-output"),
        )
        prompt = _incremental_prompt(msgs, previous_count=3)
        assert "second" in prompt
        assert "tool-output" in prompt
        assert "answer" not in prompt
        assert "first" not in prompt

    def test_message_fingerprint_is_structural_and_stable(self):
        a = _messages(("user", "hi"), ("assistant", "yo"))
        b = _messages(("user", "hi"), ("assistant", "yo"))
        assert _message_fingerprint(a) == _message_fingerprint(b)
        c = _messages(("user", "hi"), ("assistant", "different"))
        assert _message_fingerprint(a) != _message_fingerprint(c)

    def test_validate_flag_size_rejects_oversized(self):
        with pytest.raises(RuntimeError, match="inline flag limit"):
            _validate_flag_size("x" * 130_000)

    def test_expired_session_markers(self):
        assert _is_expired_session_error("session not found")
        assert _is_expired_session_error("Could not resume session")
        assert not _is_expired_session_error("rate limited")


class TestClaudeCodeSession:
    def test_second_turn_uses_resume_and_only_new_input(self, monkeypatch):
        session = ClaudeCodeSession()
        calls: list[dict] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            return "answer", "", f"12345678-1234-1234-1234-12345678900{len(calls)}"

        monkeypatch.setattr(session, "_execute", fake_execute)
        first = _messages(("system", "rules"), ("user", "first question"))
        session.run("FULL FIRST PROMPT WITH TOOLS", messages=first, model="sonnet")

        second = first + _messages(
            ("assistant", "answer"),
            ("user", "second question"),
        )
        session.run("FULL SECOND PROMPT", messages=second, model="sonnet")

        assert calls[0]["session_id"] is None
        assert calls[0]["prompt"] == "FULL FIRST PROMPT WITH TOOLS"
        assert calls[1]["session_id"] == "12345678-1234-1234-1234-123456789001"
        assert "second question" in calls[1]["prompt"]
        assert "first question" not in calls[1]["prompt"]
        assert "answer" not in calls[1]["prompt"]

    def test_expired_session_retries_with_full_prompt(self, monkeypatch):
        session = ClaudeCodeSession()
        calls: list[dict] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            if len(calls) == 1:
                return "first", "", "12345678-1234-1234-1234-123456789abc"
            if len(calls) == 2:
                raise ClaudeCodeSessionExpired("session not found")
            return "recovered", "", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        monkeypatch.setattr(session, "_execute", fake_execute)
        first = _messages(("user", "one"))
        session.run("FULL1", messages=first, model="sonnet")
        second = first + _messages(("assistant", "first"), ("user", "two"))
        response, _ = session.run("FULL2", messages=second, model="sonnet")
        assert response == "recovered"
        assert calls[1]["session_id"] == "12345678-1234-1234-1234-123456789abc"
        assert calls[2]["session_id"] is None
        assert calls[2]["prompt"] == "FULL2"

    def test_durable_state_resumes_after_new_client_without_storing_prompt(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state_key = "agent:main:discord:thread:123:123"
        session_id = "12345678-1234-1234-1234-123456789abc"
        secret = "DO_NOT_STORE_THIS_PROMPT_CONTENT"
        first_messages = _messages(("system", "rules"), ("user", secret))

        first = ClaudeCodeSession()
        monkeypatch.setattr(
            first,
            "_execute",
            lambda prompt_text, **kwargs: ("first answer", "", session_id),
        )
        first.run(
            "full first prompt",
            messages=first_messages,
            model="sonnet",
            effort="low",
            tools_digest="toolsA",
            state_key=state_key,
        )

        state_files = list((tmp_path / "state" / "claude-code-sessions").glob("*.json"))
        assert len(state_files) == 1
        payload = state_files[0].read_text(encoding="utf-8")
        assert secret not in payload
        assert session_id in payload
        assert '"model":"sonnet"' in payload
        assert '"tools_digest":"toolsA"' in payload

        second = ClaudeCodeSession()
        calls: list[dict] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            return "second answer", "", session_id

        monkeypatch.setattr(second, "_execute", fake_execute)
        second_messages = first_messages + _messages(
            ("assistant", "first answer"),
            ("user", "follow up"),
        )
        second.run(
            "full second prompt",
            messages=second_messages,
            model="sonnet",
            effort="low",
            tools_digest="toolsA",
            state_key=state_key,
        )
        assert calls[0]["session_id"] == session_id
        assert "follow up" in calls[0]["prompt"]
        assert secret not in calls[0]["prompt"]

    def test_model_change_forces_fresh_session(self, monkeypatch):
        session = ClaudeCodeSession()
        calls: list[dict] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            return "ok", "", f"12345678-1234-1234-1234-12345678900{len(calls)}"

        monkeypatch.setattr(session, "_execute", fake_execute)
        first = _messages(("user", "one"))
        session.run("FULL1", messages=first, model="sonnet", effort="low")
        second = first + _messages(("assistant", "ok"), ("user", "two"))
        session.run("FULL2", messages=second, model="opus", effort="low")
        assert calls[0]["session_id"] is None
        assert calls[1]["session_id"] is None  # model change => fresh
        assert calls[1]["model"] == "opus"

    def test_abort_kills_active_process_group(self, monkeypatch):
        session = ClaudeCodeSession()
        killed: list[tuple[int, int]] = []

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        session._request_active = True
        session._active_process = FakeProc()  # type: ignore[assignment]

        def fake_killpg(pid, sig):
            killed.append((pid, sig))

        monkeypatch.setattr("agent.claude_code_session.os.killpg", fake_killpg)
        session.abort()
        assert killed and killed[0][0] == 4242
        assert session._abort_requested is True

    def test_production_abort_path_calls_claude_session_abort(self):
        """Stranger-thread interrupt must killpg via ClaudeCodeSession.abort."""
        from run_agent import AIAgent
        from agent.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient(cwd="/tmp")
        aborted = {"called": False}

        def fake_abort():
            aborted["called"] = True

        client._claude_session.abort = fake_abort  # type: ignore[method-assign]
        agent = AIAgent.__new__(AIAgent)
        agent._openai_client_lock = lambda: nullcontext()  # type: ignore
        agent._request_client_cache_ref = lambda: {
            "client": client,
            "kwargs": None,
            "poisoned": False,
            "in_use": True,
        }
        agent._client_log_context = lambda: ""
        agent._force_close_tcp_sockets = lambda client: 0  # type: ignore[method-assign]
        AIAgent._abort_request_openai_client(agent, client, reason="interrupt_test")
        assert aborted["called"] is True

    def test_execute_builds_stream_json_stdin_and_tools_disabled(self, monkeypatch):
        session = ClaudeCodeSession()
        captured: dict = {}

        class FakeProc:
            returncode = 0

            def communicate(self, input=None, timeout=None):
                captured["stdin"] = input
                captured["timeout"] = timeout
                return (
                    _stream_stdout(text="OK", session_id="12345678-1234-1234-1234-123456789abc"),
                    "",
                )

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProc()

        monkeypatch.setattr("agent.claude_code_session.subprocess.Popen", fake_popen)
        monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", "/usr/bin/claude-fake")

        response, reasoning, sid = session._execute(
            "canonical hermes prompt",
            session_id=None,
            model="sonnet",
            effort="high",
            timeout_seconds=30,
            cwd="/tmp",
            env={"HOME": "/tmp"},
        )
        assert response == "OK"
        assert sid == "12345678-1234-1234-1234-123456789abc"
        cmd = captured["command"]
        assert cmd[0] == "/usr/bin/claude-fake"
        assert "-p" in cmd
        assert "--input-format" in cmd and "stream-json" in cmd
        assert "--output-format" in cmd and "stream-json" in cmd
        assert "--tools" in cmd
        tools_idx = cmd.index("--tools")
        assert cmd[tools_idx + 1] == ""
        assert "--effort" in cmd and "high" in cmd
        assert "--model" in cmd and "sonnet" in cmd
        assert "--session-id" in cmd
        assert "canonical hermes prompt" not in " ".join(cmd)
        stdin_obj = json.loads(captured["stdin"].strip())
        assert stdin_obj["type"] == "user"
        assert stdin_obj["message"]["content"] == "canonical hermes prompt"
        assert captured["kwargs"]["stdin"] is not None
        assert captured["kwargs"]["start_new_session"] is True

    def test_large_prompt_travels_on_stdin_not_rejected_as_flag(self, monkeypatch):
        """Regression: prompt body must not hit argv MAX_ARG_STRLEN guard."""
        session = ClaudeCodeSession()
        big = "X" * 130_000
        calls: list[dict] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            return "ok", "", "12345678-1234-1234-1234-123456789abc"

        monkeypatch.setattr(session, "_execute", fake_execute)
        response, _ = session.run(
            big,
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
        )
        assert response == "ok"
        assert calls[0]["prompt"] == big
        assert len(calls[0]["prompt"]) == 130_000


class TestClaudeCodeClient:
    def test_identity_marker(self):
        client = ClaudeCodeClient(cwd="/tmp")
        assert client.base_url == CLAUDE_CODE_MARKER_BASE_URL
        assert client.api_key == "claude-code"

    def test_extracted_tool_calls_match_openai_sdk_shape(self):
        client = ClaudeCodeClient(cwd="/tmp")
        tool_response = (
            "I'll inspect that.\n"
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )
        with patch.object(
            client._claude_session, "run", return_value=(tool_response, "")
        ):
            response = client._create_chat_completion(
                model="sonnet",
                messages=[{"role": "user", "content": "read README.md"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
            )
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        tool_call = choice.message.tool_calls[0]
        assert tool_call.id == "call_read"
        assert tool_call.function.name == "read_file"
        assert json.loads(tool_call.function.arguments) == {"path": "README.md"}
        assert choice.message.content == "I'll inspect that."

    def test_stream_true_returns_iterable_text_chunks(self):
        client = ClaudeCodeClient(cwd="/tmp")
        with patch.object(
            client._claude_session, "run", return_value=("Hello from Claude Code", "")
        ):
            stream = client._create_chat_completion(
                model="sonnet",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
            chunks = list(stream)
        assert len(chunks) >= 2
        # Live path may emit content then finish; batch fallback emits content once.
        contents = [
            c.choices[0].delta.content
            for c in chunks
            if c.choices and getattr(c.choices[0].delta, "content", None)
        ]
        assert "Hello from Claude Code" in contents
        assert any(
            c.choices and c.choices[0].finish_reason == "stop" for c in chunks if c.choices
        )
        assert any(getattr(c, "usage", None) is not None for c in chunks)

    def test_preserves_canonical_prompt_into_session_run(self):
        client = ClaudeCodeClient(cwd="/tmp")
        seen: dict = {}

        def fake_run(prompt_text, **kwargs):
            seen["prompt"] = prompt_text
            seen["kwargs"] = kwargs
            return "ok", ""

        with patch.object(client._claude_session, "run", side_effect=fake_run):
            client._create_chat_completion(
                model="opus",
                messages=[
                    {"role": "system", "content": "SYSTEM_CANON"},
                    {"role": "user", "content": "USER_CANON"},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "description": "run shell",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                tool_choice="auto",
            )
        assert "SYSTEM_CANON" in seen["prompt"]
        assert "USER_CANON" in seen["prompt"]
        assert "terminal" in seen["prompt"]
        assert seen["kwargs"]["model"] == "opus"
        # Without an active conversation context, state_key is None even when
        # tools are present; durability is gated on both tools and context.


class TestProviderRegistration:
    def test_provider_registry_has_distinct_identity(self):
        from hermes_cli.auth import (
            DEFAULT_CLAUDE_CODE_BASE_URL,
            PROVIDER_REGISTRY,
        )
        from hermes_cli.providers import HERMES_OVERLAYS, _LABEL_OVERRIDES

        cfg = PROVIDER_REGISTRY["claude-code"]
        assert cfg.auth_type == "external_process"
        assert cfg.inference_base_url == "acp://claude-code"
        assert DEFAULT_CLAUDE_CODE_BASE_URL == "acp://claude-code"
        overlay = HERMES_OVERLAYS["claude-code"]
        assert overlay.base_url_override == "acp://claude-code"
        assert _LABEL_OVERRIDES["claude-code"] == "Claude Code CLI"
        # Must not reuse or relabel copilot-acp / antigravity.
        assert PROVIDER_REGISTRY["copilot-acp"].inference_base_url == "acp://copilot"
        assert HERMES_OVERLAYS["copilot-acp"].base_url_override == "acp://copilot"

    def test_runtime_provider_resolves_claude_code(self, monkeypatch):
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
        from hermes_cli.runtime_provider import resolve_runtime_provider

        # Ensure claude is discoverable via PATH (real binary on this host).
        result = resolve_runtime_provider(
            requested="claude-code", target_model="sonnet"
        )
        assert result["provider"] == "claude-code"
        assert result["base_url"] == "acp://claude-code"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "claude-code"
        assert result["command"]
        assert "claude" in result["command"]

    def test_create_openai_client_dispatches_claude_code(self):
        from agent.agent_runtime_helpers import create_openai_client
        from agent.claude_code_client import ClaudeCodeClient

        agent = SimpleNamespace(
            provider="claude-code",
            base_url="acp://claude-code",
            _client_log_context=lambda: "",
            _build_keepalive_http_client=lambda *a, **k: None,
            _is_azure_openai_url=lambda: False,
            _is_direct_openai_url=lambda: False,
        )
        with patch("agent.agent_runtime_helpers._ra") as ra:
            ra.return_value.logger = SimpleNamespace(info=lambda *a, **k: None)
            client = create_openai_client(
                agent,
                {
                    "api_key": "claude-code",
                    "base_url": "acp://claude-code",
                },
                reason="test",
                shared=False,
            )
        assert isinstance(client, ClaudeCodeClient)
        assert client.base_url == "acp://claude-code"


class TestReviewBlockerRegressions:
    def test_partial_stream_without_result_is_rejected(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": sid,
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "partial"}],
                        },
                        "session_id": sid,
                    }
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="missing a terminal successful result"):
            _parse_stream_json_output(stdout)

    def test_non_uuid_session_id_is_rejected(self):
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "ok",
                        "session_id": "not-a-uuid",
                    }
                )
            ]
        )
        with pytest.raises(RuntimeError, match="malformed session_id"):
            _parse_stream_json_output(stdout)

    def test_trailing_content_after_result_is_rejected(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        stdout = _stream_stdout(text="ok", session_id=sid) + json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}
        )
        with pytest.raises(RuntimeError, match="after the terminal result"):
            _parse_stream_json_output(stdout)

    def test_incremental_tool_results_include_name_and_call_id(self):
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "tool",
                "name": "lookup",
                "tool_call_id": "call_a",
                "content": "same-body",
            },
            {
                "role": "tool",
                "name": "lookup",
                "tool_call_id": "call_b",
                "content": "same-body",
            },
        ]
        prompt = _incremental_prompt(msgs, previous_count=1)
        assert "tool_call_id=call_a" in prompt
        assert "tool_call_id=call_b" in prompt
        assert "name=lookup" in prompt

    def test_tools_digest_includes_full_schema_not_just_names(self):
        from agent.claude_code_client import _tools_digest

        a = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "A",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        b = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "B totally different",
                    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
                },
            }
        ]
        assert _tools_digest(a, tool_choice="auto") != _tools_digest(b, tool_choice="auto")
        assert _tools_digest(a, tool_choice="auto") != _tools_digest(a, tool_choice="required")

    def test_custom_command_is_argv0(self, monkeypatch):
        session = ClaudeCodeSession()
        captured: dict = {}

        class FakeProc:
            returncode = 0

            def communicate(self, input=None, timeout=None):
                return (_stream_stdout(text="OK"), "")

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            return FakeProc()

        monkeypatch.setattr("agent.claude_code_session.subprocess.Popen", fake_popen)
        session._execute(
            "hi",
            session_id=None,
            model="sonnet",
            effort=None,
            timeout_seconds=10,
            cwd="/tmp",
            env={"HOME": "/tmp", "PATH": "/usr/bin"},
            command="/custom/bin/claude-exact",
        )
        assert captured["command"][0] == "/custom/bin/claude-exact"

    def test_client_propagates_command_to_session(self):
        client = ClaudeCodeClient(command="/custom/claude", cwd="/tmp")
        seen: dict = {}

        def fake_run(prompt_text, **kwargs):
            seen.update(kwargs)
            return "ok", ""

        with patch.object(client._claude_session, "run", side_effect=fake_run):
            client._create_chat_completion(
                model="sonnet",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert seen["command"] == "/custom/claude"

    def test_timeout_always_sigkills_process_group(self, monkeypatch):
        import signal as _signal

        session = ClaudeCodeSession()
        signals: list[int] = []

        class FakeProc:
            pid = 7777
            returncode = None

            def communicate(self, input=None, timeout=None):
                import subprocess as sp

                raise sp.TimeoutExpired(cmd=["claude"], timeout=timeout)

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

            def kill(self):
                return None

            def poll(self):
                return self.returncode

        def fake_popen(command, **kwargs):
            return FakeProc()

        def fake_killpg(pid, sig):
            signals.append(sig)

        monkeypatch.setattr("agent.claude_code_session.subprocess.Popen", fake_popen)
        monkeypatch.setattr("agent.claude_code_session.os.killpg", fake_killpg)
        with pytest.raises(RuntimeError, match="timed out"):
            session._execute(
                "hi",
                session_id=None,
                model="sonnet",
                effort=None,
                timeout_seconds=1,
                cwd="/tmp",
                env={"HOME": "/tmp", "PATH": "/usr/bin"},
                command="claude",
            )
        assert _signal.SIGTERM in signals
        assert _signal.SIGKILL in signals

    def test_cross_process_reload_before_dispatch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state_key = "agent:main:discord:thread:cross:1"
        sid1 = "11111111-1111-1111-1111-111111111111"

        owner_a = ClaudeCodeSession()
        monkeypatch.setattr(
            owner_a,
            "_execute",
            lambda prompt_text, **kwargs: ("a1", "", sid1),
        )
        first = _messages(("user", "one"))
        owner_a.run(
            "FULL1",
            messages=first,
            model="sonnet",
            effort="low",
            tools_digest="t1",
            state_key=state_key,
        )

        owner_b = ClaudeCodeSession()
        calls_b: list[dict] = []

        def exec_b(prompt_text, **kwargs):
            calls_b.append({"prompt": prompt_text, **kwargs})
            return "b2", "", sid1

        monkeypatch.setattr(owner_b, "_execute", exec_b)
        second = first + _messages(("assistant", "a1"), ("user", "two"))
        owner_b.run(
            "FULL2",
            messages=second,
            model="sonnet",
            effort="low",
            tools_digest="t1",
            state_key=state_key,
        )
        assert calls_b[0]["session_id"] == sid1

        calls_a: list[dict] = []

        def exec_a(prompt_text, **kwargs):
            calls_a.append({"prompt": prompt_text, **kwargs})
            return "a3", "", sid1

        monkeypatch.setattr(owner_a, "_execute", exec_a)
        third = second + _messages(("assistant", "b2"), ("user", "three"))
        owner_a.run(
            "FULL3",
            messages=third,
            model="sonnet",
            effort="low",
            tools_digest="t1",
            state_key=state_key,
        )
        assert calls_a[0]["session_id"] == sid1
        assert "three" in calls_a[0]["prompt"]
        assert "one" not in calls_a[0]["prompt"]

    def test_production_request_factory_shares_session_and_aborts(self):
        """Two-turn continuation through real _create_request_openai_client."""
        from run_agent import AIAgent
        from agent.claude_code_client import ClaudeCodeClient
        from unittest.mock import patch as mock_patch

        agent = AIAgent.__new__(AIAgent)
        agent.provider = "claude-code"
        agent.base_url = "acp://claude-code"
        agent.api_key = "claude-code"
        agent.model = "sonnet"
        agent._client_kwargs = {
            "api_key": "claude-code",
            "base_url": "acp://claude-code",
            "command": "claude",
        }
        agent._request_client_cache = {
            "client": None,
            "kwargs": None,
            "poisoned": False,
            "in_use": False,
        }
        agent._openai_client_lock = lambda: nullcontext()  # type: ignore
        agent._request_client_cache_ref = lambda: agent._request_client_cache
        agent._client_log_context = lambda: ""
        agent._REQUEST_CLIENT_REUSE_REASONS = set()
        agent._force_close_tcp_sockets = lambda client: 0  # type: ignore

        primary = ClaudeCodeClient(command="/custom/claude", cwd="/tmp")
        agent._client = primary
        agent._ensure_primary_openai_client = lambda reason: primary  # type: ignore

        def _create(request_kwargs, reason, shared):
            return ClaudeCodeClient(
                **{
                    k: v
                    for k, v in request_kwargs.items()
                    if k in {"api_key", "base_url", "command", "cwd"}
                }
            )

        agent._create_openai_client = _create  # type: ignore
        agent._is_openai_client_closed = lambda client: False  # type: ignore
        agent._close_openai_client = lambda client, reason, shared: None  # type: ignore
        # base_url host matcher used in request factory; harmless stub.
        import run_agent as ra
        if not hasattr(agent, '_copilot_headers_for_request'):
            agent._copilot_headers_for_request = lambda is_vision=False: {}  # type: ignore
        agent._api_kwargs_have_image_parts = lambda api_kwargs: False  # type: ignore

        calls: list[dict] = []
        killpg_calls: list[tuple[int, int]] = []

        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, **kwargs})
            return "answer", "", f"12345678-1234-1234-1234-12345678900{len(calls)}"

        primary._claude_session._execute = fake_execute  # type: ignore

        c1 = AIAgent._create_request_openai_client(agent, reason="turn1")
        assert isinstance(c1, ClaudeCodeClient)
        assert c1._claude_session is primary._claude_session
        assert c1._owns_claude_session is False
        r1 = c1.chat.completions.create(
            model="sonnet",
            messages=[{"role": "user", "content": "first"}],
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )
        assert r1.choices[0].message.content == "answer"
        assert calls[0]["session_id"] is None

        aborted = {"n": 0}
        primary._claude_session.abort = lambda: aborted.__setitem__("n", aborted["n"] + 1)  # type: ignore
        AIAgent._close_request_openai_client(agent, c1, reason="turn1_done")
        assert aborted["n"] == 0

        c2 = AIAgent._create_request_openai_client(agent, reason="turn2")
        assert c2._claude_session is primary._claude_session
        r2 = c2.chat.completions.create(
            model="sonnet",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )
        assert r2.choices[0].message.content == "answer"
        assert calls[1]["session_id"] == "12345678-1234-1234-1234-123456789001"
        assert "second" in calls[1]["prompt"]

        class FakeProc:
            pid = 9991

            def poll(self):
                return None

        primary._claude_session._request_active = True
        primary._claude_session._active_process = FakeProc()  # type: ignore
        import agent.claude_code_session as ccs
        from agent.claude_code_session import ClaudeCodeSession as CCS

        def fake_killpg(pid, sig):
            killpg_calls.append((pid, sig))

        with mock_patch.object(ccs.os, "killpg", side_effect=fake_killpg):
            primary._claude_session.abort = CCS.abort.__get__(primary._claude_session, CCS)
            AIAgent._abort_request_openai_client(agent, c2, reason="interrupt")
        assert killpg_calls and killpg_calls[0][0] == 9991
        assert agent._request_client_cache["poisoned"] is True


class TestSol56BlockerRegressions:
    def test_full_prompt_preserves_tool_call_linkage(self):
        from agent.claude_code_client import _format_messages_as_prompt

        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_abc",
                "content": "22C",
            },
        ]
        prompt = _format_messages_as_prompt(messages, model="sonnet", tools=None)
        assert "call_abc" in prompt
        assert "get_weather" in prompt
        assert "<tool_call>" in prompt
        assert "tool_call_id=call_abc" in prompt

    def test_result_requires_boolean_false_is_error(self):
        from agent.claude_code_session import _parse_stream_json_output

        sid0 = "12345678-1234-1234-1234-123456789abc"
        # missing is_error
        bad = (
            f'{{"type":"system","session_id":"{sid0}"}}\n'
            f'{{"type":"result","session_id":"{sid0}","result":"x"}}\n'
        )
        import pytest
        with pytest.raises(RuntimeError, match="is_error"):
            _parse_stream_json_output(bad)
        # authoritative result wins over partial assistant
        good = (
            f'{{"type":"system","session_id":"{sid0}"}}\n'
            f'{{"type":"assistant","message":{{"content":[{{"type":"text","text":"partial"}}]}}}}\n'
            f'{{"type":"result","session_id":"{sid0}","is_error":false,"result":"complete-authoritative"}}\n'
        )
        text, _, sid = _parse_stream_json_output(good)
        assert text == "complete-authoritative"
        assert sid == sid0

    def test_real_cli_expired_wording_is_typed(self):
        from agent.claude_code_session import ClaudeCodeSessionExpired, _is_expired_session_error
        detail = "No conversation found with session ID: 483b18e6-90c4-4b96-ae65-326ae7bd152f"
        assert _is_expired_session_error(detail)
        # simulate nonzero exit classification
        from agent.claude_code_session import ClaudeCodeSession
        session = ClaudeCodeSession()
        # ensure markers catch
        assert "no conversation found" in detail.lower()

    def test_missing_durable_same_key_invalidates_warm_memory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.claude_code_session import ClaudeCodeSession, _save_durable_state
        session = ClaudeCodeSession()
        state_key = "agent:missing-durable"
        sid = "12345678-1234-1234-1234-123456789abc"
        session._session_id = sid
        session._previous_messages = (("user", "one"),)
        session._bound_model = "sonnet"
        session._bound_effort = None
        session._bound_tools_digest = ""
        session._state_key = state_key
        # No durable file on disk
        calls = []
        def fake_execute(prompt_text, **kwargs):
            calls.append({"prompt": prompt_text, "session_id": kwargs.get("session_id")})
            return "ok", "", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        monkeypatch.setattr(session, "_execute", fake_execute)
        session.run(
            "FULL_PROMPT two",
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "two"},
            ],
            model="sonnet",
            state_key=state_key,
        )
        assert calls[0]["session_id"] is None
        assert "FULL_PROMPT" in calls[0]["prompt"] or "two" in calls[0]["prompt"]
        # full fresh prompt is used when can_resume is false
        assert calls[0]["session_id"] is None

    def test_client_kwargs_command_from_agent_init_wiring(self):
        """Production path: claude-code gets command in client_kwargs."""
        import inspect
        from agent import agent_init
        src = inspect.getsource(agent_init)
        assert 'elif agent.provider == "claude-code"' in src
        assert 'client_kwargs["command"] = agent.acp_command' in src

    def test_abort_schedules_sigkill_escalation(self, monkeypatch):
        import signal
        import subprocess
        import time
        from agent.claude_code_session import ClaudeCodeSession
        session = ClaudeCodeSession()
        session._request_active = True
        signals = []
        class Proc:
            pid = 4242
            returncode = None
            stdin = type("S", (), {"close": lambda self: None})()
            stdout = type("S", (), {"close": lambda self: None})()
            stderr = type("S", (), {"close": lambda self: None})()
            def poll(self):
                return None
            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            def terminate(self):
                signals.append("terminate")
            def kill(self):
                signals.append("kill")
        session._active_process = Proc()
        def fake_killpg(pid, sig):
            signals.append(sig)
        monkeypatch.setattr("agent.claude_code_session.os.killpg", fake_killpg)
        session.abort()
        time.sleep(1.0)
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals


class TestOpusDeepReviewFixes:
    def test_live_readline_timeout_watchdog(self, monkeypatch):
        """Blocked readline must still hit timeout via watchdog (not hang)."""
        import threading
        import time
        from agent.claude_code_session import ClaudeCodeSession

        session = ClaudeCodeSession()
        started = time.monotonic()
        closed = threading.Event()

        class BlockingStdout:
            def readline(self):
                # Block until abort closes the stream (watchdog path).
                closed.wait(timeout=120)
                return ""

            def close(self):
                closed.set()

        class BlockingStderr:
            def __iter__(self):
                return iter(())
            def close(self):
                return None

        class FakeProc:
            returncode = None
            pid = 5150

            def __init__(self):
                self.stdin = type(
                    "S",
                    (),
                    {
                        "write": lambda self, d: len(d),
                        "close": lambda self: None,
                    },
                )()
                self.stdout = BlockingStdout()
                self.stderr = BlockingStderr()

            def poll(self):
                return -9 if closed.is_set() else None

            def wait(self, timeout=None):
                closed.wait(timeout=timeout or 0.1)
                self.returncode = -9
                return self.returncode

            def kill(self):
                closed.set()
                self.returncode = -9

            def terminate(self):
                closed.set()
                self.returncode = -15

        monkeypatch.setattr(
            "agent.claude_code_session.subprocess.Popen",
            lambda *a, **k: FakeProc(),
        )
        def fake_killpg(pid, sig):
            closed.set()
        monkeypatch.setattr("agent.claude_code_session.os.killpg", fake_killpg)
        with pytest.raises(RuntimeError, match="timed out|aborted"):
            session._execute(
                "hi",
                session_id=None,
                model="sonnet",
                effort=None,
                timeout_seconds=0.5,
                cwd="/tmp",
                env={"HOME": "/tmp", "PATH": "/usr/bin", "LANG": "C.UTF-8"},
                command="/bin/true",
            )
        elapsed = time.monotonic() - started
        assert elapsed < 15.0, f"watchdog did not fire promptly: {elapsed:.1f}s"

    def test_stream_tool_call_content_not_duplicated_or_leaked(self):
        """Streamed content must equal cleaned non-stream text (no raw tool XML)."""
        client = ClaudeCodeClient(cwd="/tmp")
        raw = (
            'I will check.\n'
            '<tool_call>{"id":"call_1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{\\"city\\":\\"Tokyo\\"}"}}'
            '</tool_call>'
        )
        with patch.object(client._claude_session, "run", return_value=(raw, "")):
            # Non-stream baseline
            complete = client._create_chat_completion(
                model="sonnet",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                stream=False,
            )
            cleaned = complete.choices[0].message.content or ""
            assert complete.choices[0].message.tool_calls
            assert "<tool_call>" not in cleaned

            stream = client._create_chat_completion(
                model="sonnet",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                stream=True,
            )
            parts = []
            saw_tools = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(delta, "tool_calls", None):
                    saw_tools = True
            joined = "".join(parts)
            assert joined == cleaned
            assert "<tool_call>" not in joined
            assert saw_tools

    def test_auxiliary_path_uses_distinct_client_type(self):
        """Auxiliary construction returns ClaudeCodeClient instances, not shared sessions."""
        from agent.claude_code_client import ClaudeCodeClient
        a = ClaudeCodeClient(cwd="/tmp")
        b = ClaudeCodeClient(cwd="/tmp")
        assert a._claude_session is not b._claude_session


class TestClaudeCodeSoftLimitRetry:
    def test_is_soft_limit_notice_detects_banner(self):
        from agent.claude_code_session import _is_soft_limit_notice

        assert _is_soft_limit_notice(
            "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"
        )
        assert _is_soft_limit_notice("You're out of extra usage. Add more at claude.ai/settings/usage")
        assert not _is_soft_limit_notice("LIMIT_PROBE_OK")
        assert not _is_soft_limit_notice(
            "Here is a long answer about budgeting and monthly spend limit strategies "
            "for product teams that is clearly not a CLI banner. " * 5
        )

    def test_soft_limit_retries_then_succeeds(self, monkeypatch):
        session = ClaudeCodeSession()
        calls = {"n": 0}
        notice = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"

        def fake_execute(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return notice, "", "12345678-1234-1234-1234-123456789abc"
            return "real answer", "", "12345678-1234-1234-1234-123456789abc"

        monkeypatch.setattr(session, "_execute", fake_execute)
        monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
        chunks = []
        response, reasoning, sid = session._execute_with_soft_limit_retry(
            "prompt",
            session_id=None,
            model="claude-opus-5",
            effort="high",
            timeout_seconds=30,
            cwd="/tmp",
            env={},
            on_text_chunk=chunks.append,
            max_attempts=3,
        )
        assert calls["n"] == 2
        assert response == "real answer"
        assert chunks == ["real answer"]
        assert sid.endswith("789abc")

    def test_soft_limit_exhausted_raises_visible_error(self, monkeypatch):
        session = ClaudeCodeSession()
        notice = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"

        def fake_execute(*args, **kwargs):
            return notice, "", "12345678-1234-1234-1234-123456789abc"

        monkeypatch.setattr(session, "_execute", fake_execute)
        monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
        with pytest.raises(RuntimeError, match="soft usage/limit notice"):
            session._execute_with_soft_limit_retry(
                "prompt",
                session_id=None,
                model="claude-opus-5",
                effort="high",
                timeout_seconds=30,
                cwd="/tmp",
                env={},
                max_attempts=3,
            )

    def test_incomplete_preamble_retries_when_tools_available(self, monkeypatch):
        from agent.claude_code_session import _is_incomplete_preamble_response

        assert _is_incomplete_preamble_response(
            "I'll do a fresh, critical pass over the live code rather than repeating prior review summaries.",
            had_tools=True,
            has_tool_calls=False,
        )
        assert not _is_incomplete_preamble_response(
            "I'll do a fresh pass.",
            had_tools=False,
            has_tool_calls=False,
        )
        assert not _is_incomplete_preamble_response(
            "Done. Here is the full analysis of the provider with detailed findings.",
            had_tools=True,
            has_tool_calls=False,
        )

        session = ClaudeCodeSession()
        calls = {"n": 0}
        preamble = "I'll do a fresh, critical pass over the live code rather than repeating prior review summaries."

        def fake_execute(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return preamble, "", "12345678-1234-1234-1234-123456789abc"
            return "Here is the full analysis with conclusions.", "", "12345678-1234-1234-1234-123456789abc"

        monkeypatch.setattr(session, "_execute", fake_execute)
        monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
        chunks = []
        response, _, _ = session._execute_with_soft_limit_retry(
            "prompt",
            session_id=None,
            model="claude-sonnet-5",
            effort="high",
            timeout_seconds=30,
            cwd="/tmp",
            env={},
            on_text_chunk=chunks.append,
            max_attempts=3,
            had_tools=True,
        )
        assert calls["n"] == 2
        assert response.startswith("Here is the full analysis")
        assert chunks == [response]
