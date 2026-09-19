"""Claude Code bridge Hermes integration (WP4).

Covers effort/thinking plumbing from the agent's reasoning config, typed API
errors (status, subscription limits, launch failures) and how the conversation
loop handles them (fail fast, fallback reason, the Claude error kept when a
fallback fails too), the context window reported by the CLI, the
provider-scoped compression threshold, retried-attempt accounting and
keep-alive chunks in the TTFT diagnostic.

Session-level tests use the fake stream-json CLI from ``test_claude_code_wp1``
(resume semantics) or the small error-scripting fake below.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.claude_code_session as ccs
from agent.claude_code_client import (
    _completion_usage,
    _resolve_effort,
    _resolve_reasoning,
)
from agent.claude_code_session import (
    ClaudeCodeAPIError,
    ClaudeCodeLaunchError,
    ClaudeCodeRefusal,
    ClaudeCodeSession,
    ClaudeCodeSoftLimitNotice,
    ClaudeCodeUsageLimitError,
    _WARM_IDLE,
    _parse_stream_json_output,
    _parse_stream_json_usage,
    clear_usage_limit_blocks,
)
from agent.context_compressor import ContextCompressor
from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_claude_code_wp1 import (  # noqa: F401 - fake_cli is a fixture
    SID,
    _arg,
    _run,
    fake_cli,
)


@pytest.fixture(autouse=True)
def _no_usage_limit_blocks():
    clear_usage_limit_blocks()
    yield
    clear_usage_limit_blocks()


# ---------------------------------------------------------------------------
# A fake CLI that scripts API errors, rate-limit events and modelUsage
# ---------------------------------------------------------------------------

ERROR_CLI = textwrap.dedent(
    """
    import json, os, sys, uuid
    argv = sys.argv[1:]
    with open(os.environ["WP4_LOG"], "a") as fh:
        fh.write(json.dumps({"argv": argv}) + "\\n")
    if os.environ.get("WP4_REJECT_THINKING") and "--thinking" in argv:
        sys.stderr.write("error: unknown option '--thinking'\\n")
        sys.exit(1)
    # Commander names only the first unknown option it meets.
    rejected = [f for f in os.environ.get("WP4_REJECT_FLAGS", "").split(",") if f]
    for flag in argv:
        if flag in rejected:
            sys.stderr.write("error: unknown option '" + flag + "'\\n")
            sys.exit(1)

    def arg(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    def next_reply():
        script = os.environ.get("WP4_SCRIPT", "")
        if not script or not os.path.exists(script):
            return {}
        counter = script + ".n"
        n = int(open(counter).read()) if os.path.exists(counter) else 0
        with open(counter, "w") as fh:
            fh.write(str(n + 1))
        replies = json.load(open(script))
        return replies[n] if n < len(replies) else {}

    def out(obj):
        # Like Claude Code (Node), non-ASCII is written as is.
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
        sys.stdout.flush()

    sid = arg("--resume") or arg("--session-id")
    for line in sys.stdin:
        reply = next_reply()
        out({"type": "system", "subtype": "init", "session_id": sid})
        if reply.get("rate_limit_info"):
            out({"type": "rate_limit_event", "rate_limit_info": reply["rate_limit_info"],
                 "session_id": sid})
        if reply.get("error_status"):
            out({"type": "assistant", "uuid": str(uuid.uuid4()), "session_id": sid,
                 "error": "rate_limit",
                 "message": {"model": "<synthetic>",
                             "content": [{"type": "text", "text": reply["result"]}]}})
            out({"type": "result", "subtype": "success", "is_error": True,
                 "api_error_status": reply["error_status"], "result": reply["result"],
                 "session_id": sid})
            continue
        if reply.get("refusal"):
            # Claude Code 2.1.277 when the API refuses and no fallback model
            # is configured (captured from the real CLI).
            explanation = reply["refusal"]
            if reply.get("refused_text"):
                out({"type": "assistant", "uuid": str(uuid.uuid4()), "session_id": sid,
                     "message": {"id": "msg_refused", "model": "claude-opus-5",
                                 "stop_reason": None,
                                 "content": [{"type": "text", "text": reply["refused_text"]}]}})
            out({"type": "system", "subtype": "model_refusal_no_fallback",
                 "original_model": "claude-opus-5", "api_refusal_category": None,
                 "api_refusal_explanation": None, "content": "", "session_id": sid})
            out({"type": "assistant", "uuid": str(uuid.uuid4()), "session_id": sid,
                 "error": "invalid_request", "is_api_error_message": True,
                 "message": {"model": "<synthetic>", "stop_reason": "refusal",
                             "content": [{"type": "text", "text": explanation}]}})
            out({"type": "result", "subtype": "success", "is_error": True,
                 "stop_reason": "refusal", "terminal_reason": "api_error",
                 "api_error_status": None, "result": explanation, "session_id": sid,
                 "total_cost_usd": 0.000825})
            continue
        text = reply.get("text", "ok")
        out({"type": "assistant", "uuid": str(uuid.uuid4()), "session_id": sid,
             "message": {"model": reply.get("model", "claude-opus-5"),
                         "content": [{"type": "text", "text": text}]}})
        if reply.get("crash"):
            sys.stderr.write("fatal: the CLI crashed\\n")
            sys.exit(1)
        result = {"type": "result", "subtype": "success", "is_error": False,
                  "result": text, "session_id": sid,
                  "usage": {"input_tokens": 10, "output_tokens": 2}}
        if reply.get("model_usage"):
            result["modelUsage"] = reply["model_usage"]
        out(result)
    """
)


@pytest.fixture
def error_cli(tmp_path, monkeypatch):
    script = tmp_path / "error_cli.py"
    script.write_text(ERROR_CLI)
    launcher = tmp_path / "claude"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    log = tmp_path / "wp4.jsonl"
    replies = tmp_path / "wp4_replies.json"
    monkeypatch.setenv("WP4_LOG", str(log))
    monkeypatch.setenv("WP4_SCRIPT", str(replies))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")

    def spawns():
        if not log.exists():
            return []
        return [json.loads(line)["argv"] for line in log.read_text().splitlines()]

    def script_replies(*items):
        replies.write_text(json.dumps(list(items)))
        counter = replies.with_name(replies.name + ".n")
        if counter.exists():
            counter.unlink()

    yield SimpleNamespace(command=str(launcher), spawns=spawns, script=script_replies, tmp=tmp_path)
    for warm in list(_WARM_IDLE.values()):
        warm.close()


def _env4():
    keys = ("PATH", "WP4_LOG", "WP4_SCRIPT", "WP4_REJECT_THINKING", "WP4_REJECT_FLAGS")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _run4(session, cli, *, model="opus", **kwargs):
    return session.run(
        "FULL PROMPT",
        messages=[{"role": "user", "content": "hi"}],
        model=model,
        tools_digest="digest",
        timeout_seconds=30,
        cwd=str(cli.tmp),
        env=_env4(),
        state_key=None,
        command=cli.command,
        system_prompt="SYSTEM",
        **kwargs,
    )


def _history(*user_texts):
    messages = []
    for index, text in enumerate(user_texts):
        if index:
            messages.append({"role": "assistant", "content": "Fake reply."})
        messages.append({"role": "user", "content": text})
    return messages


# ---------------------------------------------------------------------------
# Effort and thinking: the agent's reasoning config reaches the CLI
# ---------------------------------------------------------------------------


def _claude_agent(**kwargs):
    from run_agent import AIAgent

    return AIAgent(
        model="claude-opus-5",
        provider="claude-code",
        base_url="acp://claude-code",
        api_key="claude-code",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"enabled": True, "effort": "high"}, ("high", None)),
        # A /reasoning session override the global config does not know.
        ({"enabled": True, "effort": "xhigh"}, ("xhigh", None)),
        ({"enabled": True, "effort": "minimal"}, ("low", None)),
        ({"enabled": False}, ("low", "disabled")),
    ],
)
def test_agent_reasoning_config_reaches_the_bridge(config, expected, monkeypatch):
    """Regression: build_api_kwargs never carried reasoning for claude-code,
    so every call ran at the global agent.reasoning_effort."""

    import hermes_cli.config as hermes_config

    # The global default must not leak into an agent call.
    monkeypatch.setattr(
        hermes_config, "load_config", lambda: {"agent": {"reasoning_effort": "medium"}}
    )
    agent = _claude_agent(reasoning_config=config)
    try:
        seen = {}

        def fake_run(prompt, **kwargs):
            seen.update(kwargs)
            return "ok", ""

        monkeypatch.setattr(agent.client._claude_session, "run", fake_run)
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["reasoning"] == config
        agent.client.chat.completions.create(**kwargs)
        assert (seen["effort"], seen.get("thinking")) == expected
    finally:
        agent.close()


def test_other_providers_get_no_reasoning_key():
    from agent.chat_completion_helpers import _add_claude_code_reasoning

    kwargs = {"model": "glm-5.3"}
    agent = SimpleNamespace(
        provider="zai", base_url="https://api.z.ai/v4", reasoning_config={"effort": "high"}
    )
    _add_claude_code_reasoning(agent, kwargs)
    assert kwargs == {"model": "glm-5.3"}


def test_calls_without_reasoning_use_the_configured_default_for_the_model(monkeypatch):
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {"claude-opus-5": "high"},
            }
        },
    )
    assert _resolve_reasoning({}, "claude-opus-5") == ("high", None)
    assert _resolve_reasoning({}, "claude-sonnet-5") == ("medium", None)
    assert _resolve_effort("claude-opus-5") == "high"
    # An agent's own config wins even when it names no effort.
    assert _resolve_reasoning({"reasoning": {"enabled": True}}, "claude-opus-5") == (None, None)
    # OpenAI-style request fields.
    assert _resolve_reasoning({"reasoning_effort": "none"}, "claude-opus-5") == ("low", "disabled")
    assert _resolve_reasoning({"extra_body": {"reasoning": {"effort": "max"}}}) == ("max", None)
    monkeypatch.setattr(
        hermes_config, "load_config", lambda: {"agent": {"reasoning_effort": False}}
    )
    assert _resolve_reasoning({}, "claude-opus-5") == ("low", "disabled")


def test_reasoning_off_sends_thinking_disabled_and_keeps_the_session(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history("hi"), effort="low", thinking="disabled")
    _run(session, fake_cli, _history("hi", "again"), effort="low", thinking="disabled")
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 1  # same flags: the warm process serves the second call
    argv = spawns[0]["argv"]
    assert argv[argv.index("--thinking") + 1] == "disabled"

    # Reasoning back on: a new process (flags are fixed at spawn), same session.
    _run(session, fake_cli, _history("hi", "again", "think now"), effort="high")
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2
    assert "--thinking" not in spawns[1]["argv"]
    assert _arg(spawns[1]["argv"], "--effort") == "high"
    assert _arg(spawns[1]["argv"], "--resume") == _arg(spawns[0]["argv"], "--session-id")
    assert fake_cli.events("turn")[2]["content"] == "User:\nthink now"
    session.shutdown()


def test_effort_change_resumes_the_durable_session(fake_cli, monkeypatch, caplog):
    """Effort is a per-invocation flag, not transcript state: a /reasoning
    change must not replay the whole conversation in a fresh session."""

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    _run(ClaudeCodeSession(), fake_cli, _history("hi"), effort="medium")
    with caplog.at_level("INFO", logger="agent.claude_code_session"):
        _run(ClaudeCodeSession(), fake_cli, _history("hi", "harder"), effort="high")
    spawns = fake_cli.events("spawn")
    assert _arg(spawns[1]["argv"], "--resume") == _arg(spawns[0]["argv"], "--session-id")
    assert "--resume-session-at" in spawns[1]["argv"]
    assert _arg(spawns[1]["argv"], "--effort") == "high"
    assert fake_cli.events("turn")[1]["content"] == "User:\nharder"
    assert "resume skipped" not in caplog.text


def test_cli_rejecting_thinking_is_retried_without_it(error_cli, monkeypatch):
    monkeypatch.setattr(ccs, "_thinking_mode_supported", True)
    monkeypatch.setenv("WP4_REJECT_THINKING", "1")
    response, _reasoning = _run4(ClaudeCodeSession(), error_cli, thinking="disabled")
    assert response == "ok"
    spawns = error_cli.spawns()
    assert len(spawns) == 2
    assert "--thinking" in spawns[0] and "--thinking" not in spawns[1]
    assert ccs._thinking_mode_supported is False


def test_only_a_rejected_thinking_flag_turns_it_off(monkeypatch):
    monkeypatch.setattr(ccs, "_thinking_mode_supported", True)
    session = ClaudeCodeSession()
    with pytest.raises(ccs._CliFlagRejected):
        session._raise_process_failure(
            1, "", "error: option '--thinking <mode>' argument 'off' is invalid. "
            "Allowed choices are enabled, adaptive, disabled.", None,
        )
    assert ccs._thinking_mode_supported is False
    monkeypatch.setattr(ccs, "_thinking_mode_supported", True)
    with pytest.raises(RuntimeError):
        session._raise_process_failure(1, "", "error: unknown option '--frobnicate'", None)
    assert ccs._thinking_mode_supported is True


# ---------------------------------------------------------------------------
# API errors carry their status; launch failures are not retried
# ---------------------------------------------------------------------------


def _result_stdout(**result):
    events = [
        {"type": "system", "subtype": "init", "session_id": SID},
        dict({"type": "result", "subtype": "success", "session_id": SID}, **result),
    ]
    return "\n".join(json.dumps(event) for event in events)


@pytest.mark.parametrize(
    "status, reason",
    [
        (500, FailoverReason.server_error),
        (529, FailoverReason.overloaded),
        (401, FailoverReason.auth),
    ],
)
def test_result_api_error_status_reaches_the_classifier(status, reason):
    """Regression: every Claude Code API error was a bare RuntimeError that
    classified as ``unknown``."""

    stdout = _result_stdout(is_error=True, api_error_status=status, result=f"API Error: {status}")
    with pytest.raises(ClaudeCodeAPIError) as info:
        _parse_stream_json_output(stdout)
    assert info.value.status_code == status
    assert info.value.response.json()["error"]["message"] == f"API Error: {status}"
    assert classify_api_error(info.value, provider="claude-code", model="opus").reason == reason


def test_crashed_turn_keeps_the_result_status():
    stdout = json.dumps(
        {"type": "result", "is_error": True, "api_error_status": 401,
         "result": "OAuth token has expired", "session_id": SID}
    )
    with pytest.raises(ClaudeCodeAPIError) as info:
        ClaudeCodeSession()._raise_process_failure(1, stdout, "", None)
    assert info.value.status_code == 401
    assert classify_api_error(info.value, provider="claude-code").is_auth


# ---------------------------------------------------------------------------
# Safety refusals (Claude Code 2.1.277, API stop reason "refusal")
# ---------------------------------------------------------------------------

REFUSAL_TEXT = (
    "API Error: Opus 5's safeguards flagged this message "
    "(https://www.anthropic.com/legal/aup). This sometimes happens with safe, normal "
    "conversations. Claude Code can't respond to this message with Opus 5.\n\n"
    "Try rephrasing the request in a new session or change your model."
)

# The real CLI's events for a refused reply, trimmed of unrelated fields.
REAL_REFUSAL_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": SID, "model": "claude-opus-5"},
    {"type": "stream_event", "session_id": SID, "event": {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "REFUSEDTEXT some partial answer"}}},
    {"type": "assistant", "session_id": SID, "uuid": "287bb463-3411-46cf-987f-77c45220959a",
     "message": {"id": "msg_b570f97256f846dca184", "model": "claude-opus-5", "stop_reason": None,
                 "content": [{"type": "text", "text": "REFUSEDTEXT some partial answer"}]}},
    {"type": "system", "subtype": "model_refusal_no_fallback", "session_id": SID,
     "original_model": "claude-opus-5", "request_id": None, "api_refusal_category": None,
     "api_refusal_explanation": None, "content": "",
     "refused_user_message_uuid": "892bd1c0-1b05-42ad-aec8-ceeafb1d88f6"},
    {"type": "assistant", "session_id": SID, "uuid": "739b1aab-0734-4b3a-a692-a0c58246c7b6",
     "error": "invalid_request", "is_api_error_message": True,
     "message": {"id": "81dafded-3923-442a-bfa3-e58212fb675f", "model": "<synthetic>",
                 "stop_reason": "refusal", "content": [{"type": "text", "text": REFUSAL_TEXT}]}},
    {"type": "stream_event", "session_id": SID, "event": {
        "type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": None}}},
    {"type": "result", "subtype": "success", "is_error": True, "stop_reason": "refusal",
     "terminal_reason": "api_error", "api_error_status": None, "result": REFUSAL_TEXT,
     "session_id": SID, "total_cost_usd": 0.000825, "num_turns": 1},
]
REAL_REFUSAL_STDOUT = "\n".join(json.dumps(event) for event in REAL_REFUSAL_EVENTS)


def _assert_refusal_classification(error):
    classified = classify_api_error(error, provider="claude-code", model="claude-opus-5")
    assert classified.reason == FailoverReason.content_policy_blocked
    assert classified.retryable is False and classified.should_fallback is True


def test_real_refusal_result_is_a_refusal_not_a_retryable_error():
    """Regression (f1): the real shape (is_error, subtype "success", no API
    status) raised a generic error that Hermes retried as ``unknown``."""

    with pytest.raises(ClaudeCodeRefusal) as info:
        _parse_stream_json_output(REAL_REFUSAL_STDOUT)
    assert "safeguards flagged this message" in info.value.detail
    assert info.value.response.json()["error"]["type"] == "refusal"
    _assert_refusal_classification(info.value)


def test_refusal_on_a_failed_exit_is_a_refusal():
    with pytest.raises(ClaudeCodeRefusal) as info:
        ClaudeCodeSession()._raise_process_failure(1, REAL_REFUSAL_STDOUT, "", SID)
    assert "safeguards flagged this message" in info.value.detail
    _assert_refusal_classification(info.value)


def test_refusal_event_alone_marks_the_failure_as_a_refusal():
    events = [e for e in REAL_REFUSAL_EVENTS if e["type"] != "result"]
    events.append({"type": "result", "subtype": "success", "is_error": True,
                   "result": REFUSAL_TEXT, "session_id": SID})
    with pytest.raises(ClaudeCodeRefusal):
        _parse_stream_json_output("\n".join(json.dumps(e) for e in events))


def test_refusal_answered_by_the_cli_fallback_model_is_an_answer():
    events = [
        {"type": "system", "subtype": "init", "session_id": SID},
        {"type": "system", "subtype": "model_refusal_fallback", "session_id": SID,
         "original_model": "claude-opus-5", "fallback_model": "claude-opus-4-8"},
        {"type": "result", "subtype": "success", "is_error": False,
         "result": "The fallback's answer.", "session_id": SID, "stop_reason": "end_turn"},
    ]
    response, _reasoning, _sid = _parse_stream_json_output(
        "\n".join(json.dumps(e) for e in events)
    )
    assert response == "The fallback's answer."


def test_refusal_is_not_retried_by_the_bridge(error_cli):
    error_cli.script({"refusal": REFUSAL_TEXT, "refused_text": "REFUSEDTEXT partial"}, {})
    session = ClaudeCodeSession()
    with pytest.raises(ClaudeCodeRefusal):
        _run4(session, error_cli)
    assert len(error_cli.spawns()) == 1


def test_refusal_ends_the_turn_with_the_explanation(monkeypatch):
    """Hermes neither retries a refusal nor treats it as a cut-off reply."""

    refusal = ccs._refusal_error(REAL_REFUSAL_STDOUT, REAL_REFUSAL_EVENTS[-1])
    agent = _claude_agent()
    try:
        result, _statuses, _reasons, calls = _drive(agent, refusal, monkeypatch)
    finally:
        agent.close()
    assert calls == 1
    assert result["error"].startswith("content_policy_blocked")
    assert "safeguards flagged this message" in result["final_response"]
    assert "truncated" not in result["final_response"].lower()


def test_refusal_falls_back_once(monkeypatch):
    refusal = ccs._refusal_error(REAL_REFUSAL_STDOUT, REAL_REFUSAL_EVENTS[-1])
    agent = _claude_agent(fallback_model=[{"provider": "zai", "model": "glm-5.3"}])
    try:
        _result, _statuses, reasons, calls = _drive(agent, refusal, monkeypatch)
    finally:
        agent.close()
    assert calls == 1
    assert reasons == [FailoverReason.content_policy_blocked]


def test_refusal_after_streamed_text_ends_the_stream_as_a_refusal(monkeypatch):
    """With a live display, part of the refused reply is already on screen:
    the stream ends as ``content_filter`` with the CLI's explanation instead
    of an error Hermes would continue as a truncated reply."""

    from agent.claude_code_client import ClaudeCodeClient

    client = ClaudeCodeClient(cwd="/tmp")
    refusal = ccs._refusal_error(REAL_REFUSAL_STDOUT, REAL_REFUSAL_EVENTS[-1])

    def fake_run(prompt, *, on_text_chunk=None, **kwargs):
        on_text_chunk("REFUSEDTEXT " * 40)
        raise refusal

    monkeypatch.setattr(client._claude_session, "run", fake_run)
    chunks = list(
        client._create_chat_completion(
            model="opus", messages=[{"role": "user", "content": "q"}], stream=True
        )
    )
    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finish == ["content_filter"]
    assert content.endswith(ccs._STREAM_REFUSED_SEPARATOR + REFUSAL_TEXT)
    client.close()


def test_refusal_before_any_streamed_text_is_raised(monkeypatch):
    from agent.claude_code_client import ClaudeCodeClient

    client = ClaudeCodeClient(cwd="/tmp")
    refusal = ccs._refusal_error(REAL_REFUSAL_STDOUT, REAL_REFUSAL_EVENTS[-1])

    def fake_run(prompt, **kwargs):
        raise refusal

    monkeypatch.setattr(client._claude_session, "run", fake_run)
    with pytest.raises(ClaudeCodeRefusal):
        list(client._create_chat_completion(
            model="opus", messages=[{"role": "user", "content": "q"}], stream=True
        ))
    client.close()


def test_unlaunchable_cli_fails_fast_with_a_launch_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    not_executable = tmp_path / "claude-644"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(0o644)
    cases = [
        (str(tmp_path / "gone" / "claude"), str(tmp_path)),
        (str(not_executable), str(tmp_path)),
        (sys.executable, str(tmp_path / "no-such-dir")),
    ]
    for command, cwd in cases:
        with pytest.raises(ClaudeCodeLaunchError) as info:
            ClaudeCodeSession().run(
                "P", messages=[{"role": "user", "content": "hi"}], model="opus",
                command=command, cwd=cwd, env={}, timeout_seconds=5,
            )
        # Not an OSError: the classifier treats those as transient transport.
        assert not isinstance(info.value, OSError)
        classified = classify_api_error(info.value, provider="claude-code", model="opus")
        assert classified.reason == FailoverReason.provider_unavailable
        assert classified.retryable is False and classified.should_fallback is True
    assert "working directory" in str(info.value)


# ---------------------------------------------------------------------------
# Subscription limits
# ---------------------------------------------------------------------------


def test_usage_limit_is_not_retried_and_later_calls_fail_fast(error_cli):
    resets = int(time.time()) + 2 * 3600 + 600
    error_cli.script(
        {
            "rate_limit_info": {
                "status": "rejected", "rateLimitType": "five_hour", "resetsAt": resets,
                "isUsingOverage": False, "overageStatus": "rejected",
                "unifiedWindows": {"five_hour": {"utilization": 1.0, "resetsAt": resets}},
            },
            "error_status": 429,
            "result": "You've hit your session limit · resets 3pm (UTC)",
        },
    )
    started = time.monotonic()
    with pytest.raises(ClaudeCodeUsageLimitError) as info:
        _run4(ClaudeCodeSession(), error_cli)
    assert time.monotonic() - started < 1.9  # no soft-limit backoff
    assert len(error_cli.spawns()) == 1
    error = info.value
    assert error.status_code == 429 and error.rate_limit_type == "five_hour"
    assert error.resets_at == resets
    body = error.response.json()["error"]
    assert body["type"] == "usage_limit_reached"
    assert 7000 < body["resets_in_seconds"] <= 7800
    assert str(error).startswith("Claude 5-hour session limit reached — resets ")
    assert "(in ~2h " in str(error)

    classified = classify_api_error(error, provider="claude-code", model="opus")
    assert classified.reason == FailoverReason.rate_limit
    assert classified.retryable is False and classified.should_fallback is True
    assert classified.should_rotate_credential is False
    assert classified.error_context["usage_limit"] and classified.error_context["hard"]
    assert classified.error_context["resets_at"] == resets

    # Until the reset, calls for the model fail without starting the CLI.
    with pytest.raises(ClaudeCodeUsageLimitError):
        _run4(ClaudeCodeSession(), error_cli)
    assert len(error_cli.spawns()) == 1
    # Another model is not blocked.
    assert _run4(ClaudeCodeSession(), error_cli, model="haiku")[0] == "ok"
    assert len(error_cli.spawns()) == 2


def test_reset_wording_alone_is_a_hard_limit(error_cli):
    error_cli.script(
        {"error_status": 429, "result": "You've hit your limit · resets 3pm (America/New_York)"}
    )
    with pytest.raises(ClaudeCodeUsageLimitError) as info:
        _run4(ClaudeCodeSession(), error_cli)
    assert len(error_cli.spawns()) == 1
    assert info.value.resets_at is None
    assert "You've hit your limit · resets 3pm (America/New_York)" in str(info.value)


def test_spend_limit_banner_stays_soft_and_ends_as_a_rate_limit(monkeypatch):
    """A limit that is not exhausted until a reset is retried in-session; once
    the retries are spent Hermes gets a 429 it does not retry again."""

    session = ClaudeCodeSession()
    session._last_rate_limit = {"status": "allowed"}
    calls = []
    banner = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"

    def fake_execute(prompt, **kwargs):
        calls.append(kwargs)
        raise ClaudeCodeSoftLimitNotice(
            f"Claude Code rate-limit result: {banner}", status_code=429, detail=banner
        )

    monkeypatch.setattr(session, "_execute", fake_execute)
    monkeypatch.setattr(ccs.time, "sleep", lambda _seconds: None)
    with pytest.raises(ClaudeCodeAPIError) as info:
        session._execute_with_soft_limit_retry(
            "p", session_id=None, model="opus", effort=None, timeout_seconds=5,
            cwd="/tmp", env={},
        )
    assert len(calls) == 3
    assert not isinstance(info.value, ClaudeCodeUsageLimitError)
    assert info.value.status_code == 429
    classified = classify_api_error(info.value, provider="claude-code", model="opus")
    assert classified.reason == FailoverReason.rate_limit and classified.retryable is False
    assert classified.error_context == {"usage_limit": True}
    assert ccs._usage_limit_block("opus") is None


def test_limit_block_is_rechecked_and_cleared(monkeypatch):
    far = time.time() + 5 * 86400
    ccs._block_usage_limit("opus", {"status": "rejected", "resetsAt": far,
                                    "rateLimitType": "seven_day"}, "weekly")
    blocked = ccs._usage_limit_block("opus")
    assert isinstance(blocked, ClaudeCodeUsageLimitError)
    assert "weekly limit reached" in str(blocked) and "(in ~4d " in str(blocked)
    # A real call re-checks after the recheck interval, not days later.
    until = ccs._USAGE_LIMIT_BLOCKS["opus"][0]
    assert until <= time.time() + ccs._USAGE_LIMIT_RECHECK_SECONDS + 1
    monkeypatch.setattr(ccs.time, "time", lambda: until + 1)
    assert ccs._usage_limit_block("opus") is None


# ---------------------------------------------------------------------------
# Context window from the CLI
# ---------------------------------------------------------------------------


def test_context_window_comes_from_the_serving_model():
    stdout = _result_stdout(
        is_error=False,
        result="ok",
        usage={"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 150_000},
        modelUsage={
            "claude-haiku-4-5-20251001": {
                "inputTokens": 300, "contextWindow": 200_000, "maxOutputTokens": 32_000,
            },
            "claude-opus-5": {
                "inputTokens": 12, "cacheReadInputTokens": 150_000,
                "contextWindow": 1_000_000, "maxOutputTokens": 64_000,
            },
        },
    )
    usage = _parse_stream_json_usage(stdout)
    assert usage["context_window"] == 1_000_000 and usage["max_output_tokens"] == 64_000
    assert _completion_usage(usage).context_window == 1_000_000
    assert _completion_usage({}).context_window is None


def test_context_window_reaches_the_client_usage(error_cli):
    error_cli.script(
        {"model_usage": {"claude-opus-5": {"inputTokens": 50, "contextWindow": 200_000}}}
    )
    session = ClaudeCodeSession()
    _run4(session, error_cli)
    assert session.last_usage["context_window"] == 200_000


def _loop_agent(compressor, **extra):
    fields = dict(
        model="claude-opus-5",
        provider="claude-code",
        base_url="acp://claude-code",
        api_key="claude-code",
        api_mode="chat_completions",
        context_compressor=compressor,
        _config_context_length=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _compressor(context_length=1_000_000, provider="claude-code", **kwargs):
    kwargs.setdefault("provider_threshold_tokens", {"claude-code": 250_000})
    with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
        compressor = ContextCompressor(
            model="claude-opus-5", provider=provider, base_url="acp://claude-code",
            quiet_mode=True, **kwargs,
        )
        compressor.context_length  # resolve now, inside the patch
    return compressor


def test_reported_context_window_updates_the_compressor(monkeypatch):
    from agent import conversation_loop

    saved = []
    monkeypatch.setattr(
        conversation_loop, "save_context_length", lambda *args: saved.append(args)
    )
    compressor = _compressor()
    assert compressor.threshold_tokens == 250_000
    agent = _loop_agent(compressor)
    conversation_loop._sync_provider_context_window(agent, SimpleNamespace(context_window=200_000))
    assert compressor.context_length == 200_000
    assert compressor.threshold_tokens == 150_000  # 75% floor below 512K; cap does not bind
    assert saved == [("claude-opus-5", "acp://claude-code", 200_000)]
    # Unchanged, absent or configured windows are left alone.
    conversation_loop._sync_provider_context_window(agent, SimpleNamespace(context_window=200_000))
    conversation_loop._sync_provider_context_window(agent, SimpleNamespace())
    conversation_loop._sync_provider_context_window(agent, MagicMock())
    configured = _loop_agent(_compressor(), _config_context_length=1_000_000)
    configured.context_compressor.context_length = 1_000_000
    conversation_loop._sync_provider_context_window(
        configured, SimpleNamespace(context_window=200_000)
    )
    assert configured.context_compressor.context_length == 1_000_000
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# Provider-scoped compression threshold
# ---------------------------------------------------------------------------


def test_provider_cap_applies_only_while_its_provider_is_active():
    compressor = _compressor()
    assert compressor.threshold_tokens == 250_000
    compressor.update_model(
        model="glm-5.3", context_length=1_000_000, provider="zai", base_url="https://api.z.ai"
    )
    zai_threshold = compressor.threshold_tokens
    assert zai_threshold > 250_000
    compressor.update_model(
        model="claude-opus-5", context_length=1_000_000, provider="claude-code",
        base_url="acp://claude-code",
    )
    assert compressor.threshold_tokens == 250_000
    # A lower global cap wins; a non-positive provider cap disables it.
    assert _compressor(threshold_tokens_cap=100_000).threshold_tokens == 100_000
    uncapped = _compressor(provider_threshold_tokens={"claude-code": 0})
    assert uncapped.threshold_tokens == zai_threshold


def _init_agent(cfg):
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=1_000_000),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        agent = _claude_agent()
        threshold = agent.context_compressor.threshold_tokens
    agent.close()
    return agent, threshold


def test_claude_code_compacts_at_250k_by_default():
    agent, threshold = _init_agent({"agent": {}, "compression": {"threshold": 0.5}})
    assert agent.context_compressor.provider_threshold_tokens == {"claude-code": 250_000}
    assert threshold == 250_000
    agent, threshold = _init_agent(
        {"agent": {}, "compression": {"threshold": 0.5,
                                      "provider_threshold_tokens": {"claude-code": 300_000}}}
    )
    assert threshold == 300_000
    agent, threshold = _init_agent(
        {"agent": {}, "compression": {"threshold": 0.5,
                                      "provider_threshold_tokens": {"claude-code": 0}}}
    )
    assert threshold == 500_000


# ---------------------------------------------------------------------------
# Conversation loop: fail fast, fallback reason, the Claude error survives
# ---------------------------------------------------------------------------


def _usage_limit_error():
    return ccs._usage_limit_error(
        {"status": "rejected", "rateLimitType": "five_hour",
         "resetsAt": int(time.time()) + 3600},
        "You've hit your session limit · resets 3pm (UTC)",
    )


def _drive(agent, error, monkeypatch):
    client = agent.client = MagicMock()
    client.chat.completions.create.side_effect = error
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    statuses = []
    reasons = []

    def fake_fallback(reason=None):
        reasons.append(reason)
        return False

    monkeypatch.setattr(agent, "_emit_status", statuses.append)
    monkeypatch.setattr(agent, "_try_activate_fallback", fake_fallback)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")
    return result, statuses, reasons, client.chat.completions.create.call_count


def test_usage_limit_ends_the_turn_at_once_with_the_reset_time(monkeypatch):
    agent = _claude_agent()
    try:
        result, statuses, reasons, calls = _drive(agent, _usage_limit_error(), monkeypatch)
    finally:
        agent.close()
    assert calls == 1
    assert result["failed"] is True and result["failure_reason"] == "rate_limit"
    assert result["final_response"].startswith("Claude 5-hour session limit reached — resets ")
    assert any(s.startswith("⏳ Claude 5-hour session limit reached") for s in statuses)
    assert reasons == []  # no fallback configured


def test_usage_limit_falls_back_with_its_reason(monkeypatch):
    agent = _claude_agent(fallback_model=[{"provider": "zai", "model": "glm-5.3"}])
    try:
        result, _statuses, reasons, calls = _drive(agent, _usage_limit_error(), monkeypatch)
    finally:
        agent.close()
    assert reasons == [FailoverReason.rate_limit]
    assert calls == 1
    assert "5-hour session limit reached" in result["final_response"]


class _BillingError(Exception):
    status_code = 402

    def __init__(self):
        super().__init__("Error code: 402 - Insufficient balance or no resource package")
        self.body = {"error": {"message": "Insufficient balance or no resource package"}}
        self.response = SimpleNamespace(headers={})


def test_a_failing_fallback_does_not_hide_the_claude_limit(monkeypatch):
    """Regression: a Claude plan limit fell back to zai, whose own balance
    error was all the user saw."""

    agent = _claude_agent(fallback_model=[{"provider": "zai", "model": "glm-5.3"}])
    client = agent.client = MagicMock()
    client.chat.completions.create.side_effect = _usage_limit_error()
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    switched = []

    def fake_fallback(reason=None):
        if switched:
            return False
        switched.append(reason)
        agent.provider, agent.model = "zai", "glm-5.3"
        client.chat.completions.create.side_effect = _BillingError()
        return True

    monkeypatch.setattr(agent, "_try_activate_fallback", fake_fallback)
    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")
    finally:
        agent.close()
    assert switched == [FailoverReason.rate_limit]
    final = result["final_response"]
    assert final.startswith("Claude 5-hour session limit reached — resets ")
    assert "Fallback zai/glm-5.3 failed too: " in final
    assert "Insufficient balance" in final


def test_launch_error_is_not_retried_and_names_its_reason(monkeypatch):
    agent = _claude_agent(fallback_model=[{"provider": "zai", "model": "glm-5.3"}])
    error = ClaudeCodeLaunchError(
        "Claude Code CLI not launchable at '/x/claude' (No such file or directory). "
        "Install Claude Code or set HERMES_CLAUDE_CODE_COMMAND."
    )
    try:
        result, statuses, reasons, calls = _drive(agent, error, monkeypatch)
    finally:
        agent.close()
    assert calls == 1
    assert reasons == [FailoverReason.provider_unavailable]
    assert "not launchable at '/x/claude'" in result["final_response"]
    assert any("Non-retryable error (provider_unavailable)" in s for s in statuses)


def test_fallback_failure_reports_the_claude_error_first():
    from agent.conversation_loop import _remember_primary_failure, _with_primary_failure
    from agent.turn_retry_state import TurnRetryState

    state = TurnRetryState()
    agent = SimpleNamespace(provider="claude-code", model="claude-opus-5")
    _remember_primary_failure(agent, state, "Claude 5-hour session limit reached — resets 15:00")
    _remember_primary_failure(agent, state, "a later error")
    assert state.primary_failure_notice.startswith("Claude 5-hour")
    # Still on Claude: nothing to prepend.
    assert _with_primary_failure(agent, state, "boom") == "boom"
    agent.provider, agent.model = "zai", "glm-5.3"
    assert _with_primary_failure(agent, state, "Insufficient balance") == (
        "Claude 5-hour session limit reached — resets 15:00\n\n"
        "Fallback zai/glm-5.3 failed too: Insufficient balance"
    )
    # Only a Claude Code failure is remembered.
    other = TurnRetryState()
    _remember_primary_failure(agent, other, "zai error")
    assert other.primary_failure_notice == ""


def test_error_label_without_a_status_names_the_reason():
    from agent.conversation_loop import _error_label

    classified = SimpleNamespace(reason=FailoverReason.provider_unavailable)
    assert _error_label(None, classified) == "provider_unavailable"
    assert _error_label(401, classified) == "HTTP 401"


# ---------------------------------------------------------------------------
# Accounting and stream diagnostics
# ---------------------------------------------------------------------------


def test_retried_attempts_are_counted_but_not_as_context(monkeypatch):
    from agent import conversation_loop

    agent = SimpleNamespace(
        model="claude-opus-5", provider="claude-code", base_url="acp://claude-code",
        api_key="claude-code", api_mode="chat_completions", session_id="s1",
        _session_db=MagicMock(),
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_api_calls=1, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
    )
    usage = _completion_usage(
        {"prompt_tokens": 100, "completion_tokens": 5},
        retry={"input_tokens": 40, "cached_tokens": 60, "prompt_tokens": 100,
               "output_tokens": 7, "completion_tokens": 7, "total_tokens": 107,
               "attempts": 2},
    )
    assert usage.prompt_tokens == 100  # the final attempt only: context size
    conversation_loop._account_retry_usage(agent, usage)
    assert agent.session_api_calls == 3
    assert agent.session_output_tokens == 7
    assert agent.session_cache_read_tokens == 60
    call = agent._session_db.queue_token_counts.call_args
    assert call.kwargs["api_call_count"] == 2 and call.kwargs["output_tokens"] == 7
    # Usage without a retry part (or a mock) changes nothing.
    conversation_loop._account_retry_usage(agent, _completion_usage({"prompt_tokens": 1}))
    conversation_loop._account_retry_usage(agent, MagicMock())
    assert agent.session_api_calls == 3


def test_keepalive_chunks_are_not_a_first_token():
    from agent.chat_completion_helpers import _is_keepalive_chunk

    def chunk(**delta):
        fields = dict(role=None, content=None, tool_calls=None,
                      reasoning_content=None, reasoning=None)
        fields.update(delta)
        finish = fields.pop("finish_reason", None)
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, delta=SimpleNamespace(**fields),
                                     finish_reason=finish)],
            usage=None,
        )

    assert _is_keepalive_chunk(chunk()) is True
    assert _is_keepalive_chunk(chunk(content="Hi")) is False
    assert _is_keepalive_chunk(chunk(reasoning_content="hmm", reasoning="hmm")) is False
    assert _is_keepalive_chunk(chunk(finish_reason="stop")) is False
    assert _is_keepalive_chunk(SimpleNamespace(choices=[], usage=SimpleNamespace())) is False


# ---------------------------------------------------------------------------
# WP4 review regressions
# ---------------------------------------------------------------------------


def test_cli_rejecting_both_optional_flags_still_answers(error_cli, monkeypatch):
    """Regression: only one respawn was allowed, so a CLI that knows neither
    --thinking nor --thinking-display failed the request after two spawns."""

    monkeypatch.setattr(ccs, "_thinking_mode_supported", True)
    monkeypatch.setattr(ccs, "_thinking_display_supported", True)
    monkeypatch.setenv("WP4_REJECT_FLAGS", "--thinking,--thinking-display")
    response, _reasoning = _run4(ClaudeCodeSession(), error_cli, thinking="disabled")
    assert response == "ok"
    spawns = error_cli.spawns()
    assert len(spawns) == 3
    assert "--thinking" not in spawns[2] and "--thinking-display" not in spawns[2]


def test_cli_for_another_platform_is_a_launch_error(tmp_path, monkeypatch):
    """Regression: ENOEXEC (a corrupt download, another platform's binary)
    escaped as a raw OSError, which classifies as a transient transport
    failure and was retried."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    garbage = tmp_path / "claude"
    garbage.write_bytes(b"\x7fELF\x00 not for this machine")
    garbage.chmod(0o755)
    with pytest.raises(ClaudeCodeLaunchError) as info:
        ClaudeCodeSession().run(
            "P", messages=[{"role": "user", "content": "hi"}], model="opus",
            command=str(garbage), cwd=str(tmp_path), env={}, timeout_seconds=5,
        )
    assert "Exec format error" in str(info.value)
    classified = classify_api_error(info.value, provider="claude-code", model="opus")
    assert classified.reason == FailoverReason.provider_unavailable


def test_transient_popen_failure_stays_retryable(tmp_path, monkeypatch):
    import errno

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    def busy(*_args, **_kwargs):
        raise OSError(errno.ETXTBSY, "Text file busy", "/usr/local/bin/claude")

    monkeypatch.setattr(ccs.subprocess, "Popen", busy)
    with pytest.raises(OSError) as info:
        ClaudeCodeSession().run(
            "P", messages=[{"role": "user", "content": "hi"}], model="opus",
            command="/usr/local/bin/claude", cwd=str(tmp_path), env={}, timeout_seconds=5,
        )
    assert not isinstance(info.value, ClaudeCodeLaunchError)


def test_crashed_turn_quoting_a_limit_banner_is_not_a_limit(error_cli):
    """Regression: a turn that died after the model quoted Claude Code's limit
    wording was read as a subscription limit (from the model's own text in
    stdout) and blocked the model for ten minutes."""

    error_cli.script(
        {"text": "That notice means: You've hit your limit · resets 3pm (UTC).", "crash": True}
    )
    with pytest.raises(RuntimeError) as info:
        _run4(ClaudeCodeSession(), error_cli)
    assert not isinstance(info.value, ClaudeCodeAPIError)
    assert "exit 1" in str(info.value) and "the CLI crashed" in str(info.value)
    assert ccs._usage_limit_block("opus") is None
    assert len(error_cli.spawns()) == 1
    # The model's words are not part of the error Hermes classifies (and
    # shows): an ordinary crash, retried, not a "Claude usage limit".
    assert "That notice means" not in str(info.value)
    classified = classify_api_error(info.value, provider="claude-code", model="opus")
    assert classified.reason == FailoverReason.unknown and classified.retryable


def test_crash_after_a_cli_banner_is_still_a_limit_notice():
    banner = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"
    events = [
        {"type": "system", "subtype": "init", "session_id": SID},
        {"type": "assistant", "session_id": SID,
         "message": {"model": "<synthetic>", "content": [{"type": "text", "text": banner}]}},
    ]
    stdout = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    with pytest.raises(ClaudeCodeSoftLimitNotice) as info:
        ClaudeCodeSession()._raise_process_failure(1, stdout, "", None)
    assert info.value.detail == banner
    # The same words from the model are not a notice.
    events[1]["message"]["model"] = "claude-opus-5"
    stdout = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    with pytest.raises(RuntimeError) as info:
        ClaudeCodeSession()._raise_process_failure(1, stdout, "", None)
    assert not isinstance(info.value, ClaudeCodeSoftLimitNotice)


def test_context_window_is_the_serving_models_not_the_largest_reader():
    """modelUsage is cumulative over the CLI session, so another model (a
    refusal fallback, an earlier turn) can hold the most input."""

    events = [
        {"type": "system", "subtype": "init", "session_id": SID},
        {"type": "assistant", "session_id": SID,
         "message": {"model": "claude-opus-5", "content": [{"type": "text", "text": "ok"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "ok",
         "session_id": SID, "usage": {"input_tokens": 1},
         "modelUsage": {
             "claude-sonnet-5": {"inputTokens": 900_000, "contextWindow": 200_000,
                                 "maxOutputTokens": 64_000},
             "claude-opus-5": {"inputTokens": 5, "contextWindow": 1_000_000,
                               "maxOutputTokens": 128_000},
         }},
    ]
    usage = _parse_stream_json_usage("\n".join(json.dumps(event) for event in events))
    assert (usage["context_window"], usage["max_output_tokens"]) == (1_000_000, 128_000)


def test_context_window_sync_ignores_other_providers(monkeypatch):
    """An OpenAI-compatible SDK usage object keeps unknown response fields; a
    ``context_window`` extra from another provider must not resize Hermes."""

    from agent import conversation_loop

    monkeypatch.setattr(conversation_loop, "save_context_length", lambda *args: None)
    compressor = _compressor(provider="zai")
    agent = _loop_agent(compressor, provider="zai", model="glm-5.3", base_url="https://api.z.ai")
    conversation_loop._sync_provider_context_window(agent, SimpleNamespace(context_window=8_192))
    assert compressor.context_length == 1_000_000


class _AuthError(Exception):
    status_code = 401

    def __init__(self):
        super().__init__("Error code: 401 - OAuth token has expired")
        self.body = {"error": {"type": "authentication_error",
                               "message": "OAuth token has expired"}}
        self.response = SimpleNamespace(headers={})


def test_a_failing_fallback_does_not_hide_an_expired_claude_login(monkeypatch):
    """Regression: after an auth failover the fallback's own error was all the
    user saw; the Claude Code login problem vanished."""

    agent = _claude_agent(fallback_model=[{"provider": "zai", "model": "glm-5.3"}])
    client = agent.client = MagicMock()
    client.chat.completions.create.side_effect = ClaudeCodeAPIError(
        "Claude Code result rejected: OAuth token has expired · Please run /login",
        status_code=401,
        body=ccs._api_error_body(401, None, "OAuth token has expired · Please run /login"),
    )
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    switched = []

    def fake_fallback(reason=None):
        if switched:
            return False
        switched.append(reason)
        agent.provider, agent.model = "zai", "glm-5.3"
        client.chat.completions.create.side_effect = _BillingError()
        return True

    monkeypatch.setattr(agent, "_try_activate_fallback", fake_fallback)
    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")
    finally:
        agent.close()
    assert switched == [FailoverReason.auth]
    final = result["final_response"]
    assert final.startswith("Claude Code failed: ")
    assert "Please run /login" in final
    assert "Fallback zai/glm-5.3 failed too: " in final


def test_hard_limit_on_a_warm_streaming_turn(error_cli, monkeypatch):
    """The live path: nothing of the banner reaches the user, the process is
    not parked, and the error is the typed limit."""

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")
    resets = int(time.time()) + 3600
    error_cli.script(
        {
            "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour",
                                "resetsAt": resets, "overageStatus": "rejected"},
            "error_status": 429,
            "result": "You've hit your session limit · resets 3pm (UTC)",
        },
    )
    chunks = []
    session = ClaudeCodeSession()
    with pytest.raises(ClaudeCodeUsageLimitError):
        _run4(session, error_cli, keepalive=True, on_text_chunk=chunks.append)
    assert chunks == []
    assert session._warm is None
    assert len(error_cli.spawns()) == 1
