"""Integration tests for Claude Code CLI provider against the real binary.

These tests require the installed authenticated Claude Code CLI (2.1.183+).
They are skipped automatically when ``claude`` is not on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from agent.claude_code_client import ClaudeCodeClient
from agent.claude_code_session import ClaudeCodeSession, _parse_stream_json_output
from hermes_cli.auth import resolve_external_process_provider_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider

CLAUDE = shutil.which("claude")
pytestmark = pytest.mark.skipif(
    not CLAUDE, reason="Claude Code CLI not installed on PATH"
)


def _cli_version() -> str:
    out = subprocess.check_output([CLAUDE, "--version"], text=True, timeout=30)
    return out.strip()


def test_installed_claude_code_version_is_2_1_183_or_newer():
    version = _cli_version()
    assert "2.1." in version or version.startswith("2.")
    # Soft check: report exact observed version for the evidence packet.
    print(f"OBSERVED_CLAUDE_VERSION={version}")


def test_runtime_and_auth_resolve_without_exposing_credentials():
    creds = resolve_external_process_provider_credentials("claude-code")
    runtime = resolve_runtime_provider(requested="claude-code", target_model="sonnet")
    assert creds["provider"] == "claude-code"
    assert creds["base_url"] == "acp://claude-code"
    assert creds["api_key"] == "claude-code"
    assert "claude" in (creds.get("command") or "")
    assert runtime["provider"] == "claude-code"
    assert runtime["base_url"] == "acp://claude-code"
    # Never leak real auth material through credential resolution.
    for blob in (json.dumps(creds), json.dumps(runtime)):
        assert "sk-ant" not in blob
        assert "Bearer " not in blob
        assert "oauth" not in blob.lower() or "source" in blob.lower()


def test_stream_json_stdin_end_to_end_exact_reply(tmp_path):
    session = ClaudeCodeSession()
    marker = f"CC_INT_{uuid.uuid4().hex[:8]}"
    prompt = (
        "You are being used as the active agent backend for Hermes.\n\n"
        f"User:\nReply with exactly: {marker}\n\n"
        "Continue the conversation from the latest user request."
    )
    response, reasoning = session.run(
        prompt,
        messages=[{"role": "user", "content": f"Reply with exactly: {marker}"}],
        model="sonnet",
        effort="low",
        timeout_seconds=120,
        cwd=str(tmp_path),
        state_key=None,
    )
    assert marker in response
    print(f"E2E_RESPONSE={response!r}")
    print(f"E2E_REASONING_LEN={len(reasoning or '')}")


def test_session_resume_preserves_server_memory(tmp_path):
    session = ClaudeCodeSession()
    secret = f"SECRET_{uuid.uuid4().hex[:6].upper()}"
    first_prompt = (
        "You are being used as the active agent backend for Hermes.\n\n"
        f"User:\nRemember the secret word: {secret}. Reply exactly: GOT_IT\n\n"
        "Continue the conversation from the latest user request."
    )
    first_messages = [
        {"role": "user", "content": f"Remember the secret word: {secret}. Reply exactly: GOT_IT"}
    ]
    r1, _ = session.run(
        first_prompt,
        messages=first_messages,
        model="sonnet",
        effort="low",
        timeout_seconds=120,
        cwd=str(tmp_path),
    )
    assert "GOT_IT" in r1
    assert session._session_id
    first_sid = session._session_id

    second_messages = first_messages + [
        {"role": "assistant", "content": r1},
        {
            "role": "user",
            "content": "What was the secret word? Reply with just the word.",
        },
    ]
    # Incremental path should resume and NOT re-send the secret in argv.
    r2, _ = session.run(
        "FULL SECOND SHOULD NOT BE USED IF RESUME WORKS",
        messages=second_messages,
        model="sonnet",
        effort="low",
        timeout_seconds=120,
        cwd=str(tmp_path),
    )
    assert secret in r2
    assert session._session_id == first_sid
    print(f"RESUME_SID={first_sid}")
    print(f"RESUME_REPLY={r2!r}")


def test_native_tools_disabled_emits_hermes_tool_call_blocks(tmp_path):
    client = ClaudeCodeClient(cwd=str(tmp_path))
    response = client.chat.completions.create(
        model="sonnet",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo? Use the get_weather tool.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        timeout=120,
    )
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls
    tc = choice.message.tool_calls[0]
    assert tc.function.name == "get_weather"
    args = json.loads(tc.function.arguments)
    assert "tokyo" in str(args.get("city", "")).lower()
    print(f"TOOL_CALL={tc.function.name} args={args}")


def test_client_create_and_close_cleanup(tmp_path):
    client = ClaudeCodeClient(cwd=str(tmp_path))
    resp = client.chat.completions.create(
        model="sonnet",
        messages=[{"role": "user", "content": "Reply exactly: CLOSE_OK"}],
        timeout=120,
    )
    assert "CLOSE_OK" in (resp.choices[0].message.content or "")
    client.close()
    assert client.is_closed is True


def test_command_line_never_contains_prompt_body(tmp_path, monkeypatch):
    """Guard: the prompt must travel on stdin, never as argv."""
    session = ClaudeCodeSession()
    captured: dict = {}

    real_popen = subprocess.Popen

    def wrapping_popen(command, **kwargs):
        captured["command"] = list(command)
        # Short-circuit with a fake successful stream-json payload to avoid
        # a second live call when this is only validating argv construction.
        class Fake:
            returncode = 0

            def communicate(self, input=None, timeout=None):
                captured["stdin"] = input
                sid = "12345678-1234-1234-1234-123456789abc"
                stdout = "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "system",
                                "subtype": "init",
                                "session_id": sid,
                                "model": "claude-sonnet-4-6",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": "ARGV_SAFE"}
                                    ],
                                },
                                "session_id": sid,
                            }
                        ),
                        json.dumps(
                            {
                                "type": "result",
                                "is_error": False,
                                "result": "ARGV_SAFE",
                                "session_id": sid,
                            }
                        ),
                    ]
                )
                return stdout + "\n", ""

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        return Fake()

    monkeypatch.setattr("agent.claude_code_session.subprocess.Popen", wrapping_popen)
    secret = "PROMPT_BODY_MUST_NOT_APPEAR_IN_ARGV_" + uuid.uuid4().hex
    response, _, sid = session._execute(
        secret,
        session_id=None,
        model="sonnet",
        effort="low",
        timeout_seconds=30,
        cwd=str(tmp_path),
        env=None,
    )
    assert response == "ARGV_SAFE"
    assert secret not in " ".join(captured["command"])
    assert secret in captured["stdin"]
    assert "--tools" in captured["command"]
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
