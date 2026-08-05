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
        with pytest.raises(RuntimeError, match="result error"):
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
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello from Claude Code"
        assert chunks[0].choices[0].finish_reason == "stop"
        assert chunks[1].choices == []
        assert chunks[1].usage.total_tokens == 0

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
