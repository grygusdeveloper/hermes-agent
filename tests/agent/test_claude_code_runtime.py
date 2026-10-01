"""Claude Code runtime behaviour ported from the fork (WP3/WP4/WP5, f8a124c66f, ae77bcfb28).

Error semantics (refusals, expired sessions, soft and hard usage limits), plan-limit
recording and notices, CLI env/flags, quiet work turns and per-turn telemetry, driven
through a scripted stream-json CLI subprocess; conversation-loop handling of Claude
limits through a real AIAgent with only the provider call replaced.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.claude_code_events as cce
import agent.claude_code_session as ccs
from agent.claude_code_client import ClaudeCodeClient
from agent.claude_code_errors import (
    ClaudeCodeAPIError, ClaudeCodeLaunchError, ClaudeCodeRefusal, ClaudeCodeSoftLimitNotice,
    ClaudeCodeUsageLimitError,
)
from agent.claude_code_limits import (
    block_usage_limit, clear_usage_limit_blocks, rate_limit_path, read_rate_limit_snapshot,
    show_claude_code_notices, usage_limit_block, usage_limit_error,
)
from agent.claude_code_protocol import ELIDE_ARG_CHARS, system_and_transcript
from agent.error_classifier import FailoverReason, classify_api_error
from agent.portal_tags import reset_native_transport_identity, set_native_transport_identity
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

pytestmark = pytest.mark.platforms("posix")

CLI = r'''
import json, os, sys, uuid
from pathlib import Path
root = Path(__file__).parent
args = sys.argv[1:]
plan = json.loads((root / "plan.json").read_text())
log = root / "spawns.jsonl"
index = len(log.read_text().splitlines()) if log.exists() else 0
watch = ("DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING", "DISABLE_AUTO_COMPACT",
         "CLAUDE_CODE_PROMPT_CACHE_TTL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
with log.open("a") as out:
    out.write(json.dumps({"args": args, "env": {k: os.environ.get(k) for k in watch}}) + "\n")
step = plan[min(index, len(plan) - 1)]
if step.get("reject_thinking_display") and "--thinking-display" in args:
    sys.stderr.write("error: unknown option '--thinking-display'\n")
    sys.exit(1)
if "stderr_exit" in step:
    sys.stderr.write(step["stderr_exit"])
    sys.exit(1)
sid = args[args.index("--resume") + 1] if "--resume" in args else args[args.index("--session-id") + 1]
def fill(value):
    if isinstance(value, dict):
        return {k: fill(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v) for v in value]
    if value == "$SID":
        return sid
    if value == "$UUID":
        return str(uuid.uuid4())
    return value
def send(value):
    print(json.dumps(fill(value)), flush=True)
def read():
    return json.loads(sys.stdin.readline())
def control(mid, method, params=None):
    send({"type": "control_request", "request_id": str(mid), "request": {"subtype": "mcp_message",
          "server_name": "hermes", "message": {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}}})
    return read()
init = read()
if json.loads(args[args.index("--mcp-config") + 1])["mcpServers"]:
    control(1, "initialize", {"protocolVersion": "2024-11-05"})
    control(2, "tools/list")
send({"type": "control_response", "response": {"subtype": "success", "request_id": init["request_id"], "response": {}}})
# One process may serve several user turns while Hermes keeps it warm ("next").
for turn, events in enumerate([step.get("events", [])] + step.get("next", [])):
    request = read()
    (root / ("request-%d.json" % index if not turn else "request-%d-%d.json" % (index, turn))).write_text(
        json.dumps(request))
    send({"type": "system", "subtype": "init", "session_id": sid, "model": args[args.index("--model") + 1]})
    for event in events:
        if event.get("type") == "TOOL_CALL":
            send({"type": "control_request", "request_id": "perm", "request": {"subtype": "can_use_tool",
                  "tool_name": "mcp__hermes__" + event["name"], "input": event["input"], "tool_use_id": event["id"]}})
            read()
            control(3, "tools/call", {"name": event["name"], "arguments": event["input"]})
            continue
        send(event)
for line in sys.stdin:
    pass
'''

MODEL = "claude-opus-5"
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]


def partial(event):
    return {"type": "stream_event", "session_id": "$SID", "event": event}


def assistant(*blocks, stop="end_turn", model=MODEL):
    """Partial events + the authoritative assistant message, in CLI 2.1.277 order."""
    events = [partial({"type": "message_start"})]
    full = []
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "tool_use":
            start = {"type": "tool_use", "id": block["id"], "name": "mcp__hermes__" + block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            full.append({"type": "tool_use", "id": block["id"], "name": "mcp__hermes__" + block["name"],
                         "input": block["input"]})
        else:
            key = "text" if kind == "text" else "thinking"
            start = {"type": kind, key: ""}
            delta = {"type": f"{key}_delta", key: block[key]}
            full.append(block)
        events.append(partial({"type": "content_block_start", "index": index, "content_block": start}))
        events.append(partial({"type": "content_block_delta", "index": index, "delta": delta}))
        if index < len(blocks) - 1:
            events.append(partial({"type": "content_block_stop", "index": index}))
    events.append({"type": "assistant", "uuid": "$UUID", "session_id": "$SID",
                   "message": {"role": "assistant", "model": model, "content": full, "stop_reason": stop}})
    events += [partial({"type": "content_block_stop", "index": len(blocks) - 1}),
               partial({"type": "message_delta", "delta": {"stop_reason": stop}}),
               partial({"type": "message_stop"})]
    return events


def text(value):
    return {"type": "text", "text": value}


def success(result="ok", **extra):
    return {"type": "result", "subtype": "success", "is_error": False, "session_id": "$SID", "result": result,
            "usage": {"input_tokens": 5, "output_tokens": 2}, **extra}


def answer(value="Real answer."):
    return {"events": [*assistant(text(value)), success(value)]}


def synthetic(value, error=None):
    event = {"type": "assistant", "uuid": "$UUID", "session_id": "$SID",
             "message": {"role": "assistant", "model": "<synthetic>", "content": [text(value)], "stop_reason": "stop_sequence"}}
    if error:
        event["error"] = error
    return event


def failed(result, **extra):
    return {"type": "result", "subtype": "success", "is_error": True, "session_id": "$SID", "result": result, **extra}


def window_info(utilization, reset, status="allowed", weekly=None):
    windows = {"five_hour": {"utilization": utilization, "resetsAt": reset}}
    if weekly is not None:
        windows["seven_day"] = {"utilization": weekly, "resetsAt": reset + 86400}
    return {"status": status, "rateLimitType": "five_hour", "resetsAt": reset, "unifiedWindows": windows}


def rate_event(info):
    return {"type": "rate_limit_event", "rate_limit_info": info, "session_id": "$SID"}


REFUSAL_TEXT = ("API Error: Opus 5's safeguards flagged this message (https://www.anthropic.com/legal/aup). "
                "Claude Code can't respond to this message with Opus 5.")
# The real CLI's events for a refused reply (Claude Code 2.1.277), trimmed: a partial
# reply, then the refusal event, a <synthetic> explanation and an is_error result.
REFUSAL_TAIL = [
    {"type": "system", "subtype": "model_refusal_no_fallback", "session_id": "$SID",
     "original_model": MODEL, "content": ""},
    synthetic(REFUSAL_TEXT, error="invalid_request") | {"is_api_error_message": True},
    failed(REFUSAL_TEXT, stop_reason="refusal", terminal_reason="api_error", api_error_status=None),
]


def refused(partial_text):
    """``partial_text`` streams, then the refusal arrives before the block stops."""
    return [*assistant(text(partial_text), stop=None)[:-3], *REFUSAL_TAIL[:2],
            partial({"type": "content_block_stop", "index": 0}), REFUSAL_TAIL[2]]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\n" + CLI)
    executable.chmod(0o700)
    monkeypatch.setattr(ccs, "_SOFT_LIMIT_BASE_DELAY", 0.0)
    monkeypatch.setattr(ccs, "_thinking_display_supported", True)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", raising=False)
    # Each scripted step is one CLI process; warm reuse has its own tests below.
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    clear_usage_limit_blocks()
    home = set_hermes_home_override(tmp_path / "home")
    owner = set_native_transport_identity("conversation")

    class Fake:
        root = tmp_path
        command = str(executable)

        def script(self, *steps):
            (tmp_path / "plan.json").write_text(json.dumps(list(steps)))

        def spawns(self):
            log = tmp_path / "spawns.jsonl"
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        def request(self, index, turn=0):
            name = f"request-{index}.json" if not turn else f"request-{index}-{turn}.json"
            content = json.loads((tmp_path / name).read_text())["message"]["content"]
            return json.loads(content)["conversation_records"]

        def client(self):
            made = ClaudeCodeClient(command=str(executable), cwd=str(tmp_path))
            clients.append(made)
            return made

    clients = []
    try:
        yield Fake()
    finally:
        for made in clients:
            made.close()
        reset_native_transport_identity(owner)
        reset_hermes_home_override(home)
        clear_usage_limit_blocks()


def ask(client, messages=None, **kwargs):
    return client.chat.completions.create(model=kwargs.pop("model", MODEL),
                                          messages=messages or [{"role": "user", "content": "hi"}],
                                          timeout=kwargs.pop("timeout", 10), **kwargs)


def history(turns):
    rows = [{"role": "user", "content": "hi"}]
    for index in range(turns - 1):
        rows += [{"role": "assistant", "content": "Real answer."}, {"role": "user", "content": f"again {index}"}]
    return rows


def checkpoints(cli):
    return list((cli.root / "home" / "claude-code" / "checkpoints").glob("*.json"))


# ── Typed API errors ─────────────────────────────────────────────────────


@pytest.mark.parametrize("status, reason", [(500, FailoverReason.server_error), (529, FailoverReason.overloaded),
                                            (401, FailoverReason.auth)])
def test_result_api_error_status_reaches_the_classifier(cli, status, reason):
    """Regression: every Claude Code API error was a bare RuntimeError (``unknown``)."""
    cli.script({"events": [failed(f"API Error: {status}", api_error_status=status)]})
    with pytest.raises(ClaudeCodeAPIError) as info:
        ask(cli.client())
    assert info.value.status_code == status
    assert info.value.response.json()["error"]["message"] == f"API Error: {status}"
    assert classify_api_error(info.value, provider="claude-code", model=MODEL).reason == reason


def test_refusal_is_a_refusal_not_a_retryable_error(cli):
    cli.script({"events": refused("REFUSEDTEXT some partial answer")}, answer())
    with pytest.raises(ClaudeCodeRefusal) as info:
        ask(cli.client())
    assert "safeguards flagged this message" in info.value.detail
    assert info.value.response.json()["error"]["type"] == "refusal"
    classified = classify_api_error(info.value, provider="claude-code", model=MODEL)
    assert classified.reason == FailoverReason.content_policy_blocked
    assert classified.retryable is False and classified.should_fallback is True
    assert len(cli.spawns()) == 1  # never retried by the bridge


def test_refusal_after_streamed_text_ends_the_stream_as_a_refusal(cli):
    """Part of the refused reply is on screen: the stream ends as content_filter with the
    CLI's explanation, outside any code block it left open, and nothing is published."""
    cli.script(answer(), {"events": refused("Here:\n\n```python\nprint(1)\n")})
    client = cli.client()
    first = ask(client)
    before = checkpoints(cli)[0].read_bytes()
    chunks = list(ask(client, history(2), stream=True))
    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert first.choices[0].message.content == "Real answer."
    assert finish == ["content_filter"]
    assert content == "Here:\n\n```python\nprint(1)\n\n```" + cce.REFUSED_SEPARATOR + REFUSAL_TEXT
    assert checkpoints(cli)[0].read_bytes() == before


def test_refusal_before_any_streamed_text_is_raised(cli):
    cli.script({"events": REFUSAL_TAIL})
    with pytest.raises(ClaudeCodeRefusal):
        list(ask(cli.client(), stream=True))


def test_cli_model_fallback_is_a_notice_shown_at_most_every_ten_minutes(cli, monkeypatch):
    """A flagged conversation falls back on every call; regression: the fallback event
    failed the turn, and later each repeat became its own status message."""
    fallback = {"type": "system", "subtype": "model_refusal_fallback", "session_id": "$SID",
                "originalModel": MODEL, "fallbackModel": "claude-opus-4-8", "scope": "session",
                "content": "Opus 5's safeguards flagged this message. Switched to Opus 4.8."}
    step = {"events": [fallback, *assistant(text("Fallback answer."), model="claude-opus-4-8"), success()]}
    cli.script(step)
    client = cli.client()
    first = ask(client)
    assert first.choices[0].message.content == "Fallback answer."
    assert first.usage.hermes_claim_notices() == [
        "⚠️ Claude Code: Opus 5's safeguards flagged this message. Switched to Opus 4.8."]
    assert ask(client, history(2)).usage.hermes_claim_notices() == []
    real_time = time.time
    monkeypatch.setattr(ccs.time, "time", lambda: real_time() + 3600)
    assert len(ask(client, history(5)).usage.hermes_claim_notices()) == 1


# ── Expired sessions ─────────────────────────────────────────────────────


@pytest.mark.parametrize("where", ["stderr", "result"])
def test_expired_session_drops_the_checkpoint_and_replays_once(cli, where):
    gone = "No conversation found with session ID: 483b18e6-90c4-4b96-ae65-326ae7bd152f"
    expired = {"stderr_exit": gone + "\n"} if where == "stderr" else {"events": [
        {"type": "result", "subtype": "error_during_execution", "is_error": True, "session_id": "$SID",
         "errors": [gone]}]}
    cli.script(answer("First."), expired, answer("Second."))
    client = cli.client()
    ask(client)
    old = json.loads(checkpoints(cli)[0].read_text())["session_id"]
    second = ask(client, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "First."},
                          {"role": "user", "content": "more"}])
    assert second.choices[0].message.content == "Second."
    spawns = cli.spawns()
    assert len(spawns) == 3
    assert "--resume" in spawns[1]["args"] and "--resume" not in spawns[2]["args"]
    # The fresh conversation carries the whole transcript, not only the new tail.
    assert [r["text"] for r in cli.request(2)] == ["hi", "First.", "more"]
    assert json.loads(checkpoints(cli)[0].read_text())["session_id"] != old


def test_expired_wording_on_a_fresh_session_is_not_retried(cli):
    cli.script({"events": [failed("No conversation found with session ID: x")]})
    with pytest.raises(RuntimeError):
        ask(cli.client())
    assert len(cli.spawns()) == 1


# ── Soft and hard usage limits ───────────────────────────────────────────

SPEND_BANNER = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"


def soft_notice():
    return {"events": [synthetic(SPEND_BANNER, error="rate_limit"), failed(SPEND_BANNER, api_error_status=429)]}


def test_soft_usage_notice_is_retried_never_answered_with(cli):
    cli.script(soft_notice(), answer("Full real answer here."))
    response = ask(cli.client())
    assert response.choices[0].message.content == "Full real answer here."
    assert len(cli.spawns()) == 2


def test_banner_instead_of_a_successful_answer_is_retried(cli):
    """9112f82f46: the CLI's banner arrived as a *successful* result."""
    cli.script({"events": [synthetic(SPEND_BANNER), success(SPEND_BANNER)]}, answer())
    assert ask(cli.client()).choices[0].message.content == "Real answer."
    assert len(cli.spawns()) == 2


def test_soft_limit_retries_are_bounded_and_end_as_a_rate_limit(cli):
    cli.script(soft_notice())
    with pytest.raises(ClaudeCodeAPIError) as info:
        ask(cli.client())
    assert len(cli.spawns()) == 3
    assert not isinstance(info.value, (ClaudeCodeUsageLimitError, ClaudeCodeSoftLimitNotice))
    assert info.value.status_code == 429
    classified = classify_api_error(info.value, provider="claude-code", model=MODEL)
    assert classified.reason == FailoverReason.rate_limit and classified.retryable is False
    assert classified.should_fallback is True and classified.error_context == {"usage_limit": True}
    assert usage_limit_block(MODEL) is None


def test_hard_usage_limit_is_not_retried_and_later_calls_fail_fast(cli):
    resets = int(time.time()) + 2 * 3600 + 600
    info = {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": resets, "isUsingOverage": False,
            "overageStatus": "rejected", "unifiedWindows": {"five_hour": {"utilization": 1.0, "resetsAt": resets}}}
    banner = "You've hit your session limit · resets 3pm (UTC)"
    cli.script({"events": [rate_event(info), synthetic(banner, error="rate_limit"),
                           failed(banner, api_error_status=429)]}, answer())
    with pytest.raises(ClaudeCodeUsageLimitError) as raised:
        ask(cli.client())
    assert len(cli.spawns()) == 1
    error = raised.value
    assert error.status_code == 429 and error.rate_limit_type == "five_hour" and error.resets_at == resets
    body = error.response.json()["error"]
    assert body["type"] == "usage_limit_reached" and 7000 < body["resets_in_seconds"] <= 7800
    assert str(error).startswith("Claude 5-hour session limit reached — resets ") and "(in ~2h " in str(error)
    classified = classify_api_error(error, provider="claude-code", model=MODEL)
    assert classified.reason == FailoverReason.rate_limit
    assert (classified.retryable, classified.should_fallback, classified.should_rotate_credential) == (False, True, False)
    assert classified.error_context["usage_limit"] and classified.error_context["hard"]
    assert classified.error_context["resets_at"] == resets
    # Until the reset, calls for the model fail without starting the CLI ...
    with pytest.raises(ClaudeCodeUsageLimitError):
        ask(cli.client())
    assert len(cli.spawns()) == 1
    # ... while another model is not blocked.
    assert ask(cli.client(), model="haiku").choices[0].message.content == "Real answer."


def test_reset_wording_alone_is_a_hard_limit(cli):
    banner = "You've hit your limit · resets 3pm (America/New_York)"
    cli.script({"events": [failed(banner, api_error_status=429)]})
    with pytest.raises(ClaudeCodeUsageLimitError) as info:
        ask(cli.client())
    assert len(cli.spawns()) == 1 and info.value.resets_at is None
    assert banner in str(info.value)


def test_limit_block_is_rechecked_and_expires(monkeypatch):
    far = time.time() + 5 * 86400
    block_usage_limit("opus", {"status": "rejected", "resetsAt": far, "rateLimitType": "seven_day"}, "weekly")
    blocked = usage_limit_block("opus")
    assert isinstance(blocked, ClaudeCodeUsageLimitError)
    assert "weekly limit reached" in str(blocked) and "(in ~4d " in str(blocked)
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 601)
    assert usage_limit_block("opus") is None
    monkeypatch.undo()
    clear_usage_limit_blocks()


def test_pending_warning_carries_the_latest_figures_and_drops_after_its_reset(cli, monkeypatch):
    reset = time.time() + 3600
    session = ccs.ClaudeCodeSession("claude", None)
    session._record_rate_limit(window_info(0.91, reset))
    session._record_rate_limit(window_info(0.97, reset))
    notices = session.take_notices()
    assert len(notices) == 1 and "97% used" in notices[0]
    later = ccs.ClaudeCodeSession("claude", None)
    later._record_rate_limit(window_info(0.95, reset + 18000))
    # No reply carried it before the window reset (hours without a turn).
    real_time = time.time
    monkeypatch.setattr(ccs.time, "time", lambda: real_time() + 18000 + 7200)
    assert later.take_notices() == []


def test_plan_snapshot_updates_hold_a_cross_process_lock(cli):
    import fcntl

    from agent.claude_code_limits import _rate_limit_update, state_dir

    with _rate_limit_update():
        fd = os.open(state_dir() / "rate_limit.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


def test_fallback_notice_without_a_named_fallback_model():
    from agent.claude_code_errors import model_fallback_notice

    text = model_fallback_notice("model_refusal_fallback", {"originalModel": MODEL, "apiRefusalCategory": "cyber"})
    assert text == f"⚠️ {MODEL} declined this request (cyber); another model answered instead."
    assert "None" not in model_fallback_notice("model_fallback", {})


def test_model_text_quoting_a_limit_banner_is_not_a_limit(cli):
    """Only the CLI's own words decide: a model answer may discuss limits freely."""
    quote = "That notice means: You've hit your limit · resets 3pm (UTC)."
    cli.script({"events": [*assistant(text(quote)), success(quote)]})
    assert ask(cli.client()).choices[0].message.content == quote
    assert len(cli.spawns()) == 1 and usage_limit_block(MODEL) is None


# ── Plan-limit recording and one-time warnings ───────────────────────────


def test_rate_limit_event_is_persisted_in_the_existing_format(cli):
    reset = int(time.time()) + 3600
    cli.script({"events": [rate_event(window_info(0.3, reset)), *answer()["events"]]})
    response = ask(cli.client())
    snapshot = read_rate_limit_snapshot()
    assert snapshot["info"]["unifiedWindows"]["five_hour"]["utilization"] == 0.3
    assert abs(snapshot["recorded_at"] - time.time()) < 60 and snapshot["warned"] == {}
    path = rate_limit_path()
    assert path == cli.root / "home" / "state" / "claude-code-sessions" / "rate_limit.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert set(json.loads(path.read_text())) == {"info", "recorded_at", "warned"}
    assert response.usage.hermes_claim_notices() == []  # below the warning threshold


def test_plan_warning_is_announced_once_per_window_cycle(cli):
    reset = int(time.time()) + 3600
    cli.script({"events": [rate_event(window_info(0.92, reset)), *answer()["events"]]},
               {"events": [rate_event(window_info(0.95, reset)), *answer()["events"]]},
               {"events": [rate_event(window_info(0.91, reset + 5 * 3600)), *answer()["events"]]})
    client = cli.client()
    notices = ask(client).usage.hermes_claim_notices()
    assert len(notices) == 1 and "5-hour session limit is 92% used" in notices[0] and "resets" in notices[0]
    assert read_rate_limit_snapshot()["warned"] == {"five_hour": reset}
    assert ask(client, history(2)).usage.hermes_claim_notices() == []  # same window cycle
    # Another session (another agent, a restarted gateway) remembers the announcement.
    other = ccs.ClaudeCodeSession("claude", None)
    other._record_rate_limit(window_info(0.97, reset))
    assert other.take_notices() == []
    assert [n for n in ask(client, history(3)).usage.hermes_claim_notices() if "91% used" in n]


def test_warning_seen_first_by_an_auxiliary_call_is_not_swallowed(cli):
    reset = int(time.time()) + 3600
    hot = window_info(0.94, reset)
    cli.script({"events": [rate_event(hot), *answer()["events"]]})
    aux = ClaudeCodeClient(command=cli.command, cwd=str(cli.root), auxiliary=True)
    try:
        assert getattr(ask(aux).usage, "hermes_claim_notices", None) is None
    finally:
        aux.close()
    main = ccs.ClaudeCodeSession("claude", None)
    main._record_rate_limit(hot)
    assert len(main.take_notices()) == 1


def test_cli_warning_status_is_announced_and_rejections_and_reset_windows_are_not(cli):
    reset = time.time() + 7200
    session = ccs.ClaudeCodeSession("claude", None)
    session._record_rate_limit({"status": "allowed_warning", "rateLimitType": "seven_day", "resetsAt": reset})
    notices = session.take_notices()
    assert len(notices) == 1 and "weekly limit is almost used up" in notices[0]
    session._record_rate_limit(window_info(1.0, reset + 60, status="rejected"))
    assert session.take_notices() == []
    session._record_rate_limit(window_info(0.99, time.time() - 10))
    assert session.take_notices() == []


def test_weekly_twin_windows_make_one_notice_per_cycle(cli):
    reset = int(time.time()) + 3 * 86400
    twins = {"status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.95, "resetsAt": reset},
             "seven_day_overage_included": {"utilization": 0.95, "resetsAt": reset}}}
    session = ccs.ClaudeCodeSession("claude", None)
    session._record_rate_limit(twins)
    first = session.take_notices()
    assert len(first) == 1 and "weekly limit is 95% used" in first[0]
    other = ccs.ClaudeCodeSession("claude", None)
    other._record_rate_limit(twins)
    assert other.take_notices() == []


def test_notices_are_shown_only_for_claude_code_status_surfaces():
    shown, claims = [], []

    def claim():
        claims.append(1)
        return ["⚠️ Claude weekly limit is 95% used.", "", 3]

    class Agent:
        def __init__(self, provider, platform="discord", callback=True):
            self.provider, self.platform = provider, platform
            self.status_callback = (lambda *a: None) if callback else None
            self._emit_status = shown.append

    usage = type("Usage", (), {"hermes_claim_notices": staticmethod(claim)})()
    show_claude_code_notices(Agent("claude-code"), usage)
    assert shown == ["⚠️ Claude weekly limit is 95% used."]
    show_claude_code_notices(Agent("openrouter"), usage)
    # A subagent's status lines reach no one: it never claims (the warning stays pending).
    show_claude_code_notices(Agent("claude-code", platform="subagent", callback=False), usage)
    assert len(shown) == 1 and len(claims) == 1
    show_claude_code_notices(Agent("claude-code", platform="cli", callback=False), usage)
    assert len(shown) == 2


# ── CLI flags, env, launch errors and telemetry ──────────────────────────


def test_thinking_display_and_child_env_defaults(cli, monkeypatch):
    monkeypatch.setenv("DISABLE_TELEMETRY", "0")
    cli.script(answer())
    ask(cli.client())
    spawn = cli.spawns()[0]
    assert spawn["args"][spawn["args"].index("--thinking-display") + 1] == "summarized"
    assert spawn["env"] == {"DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "0", "DISABLE_ERROR_REPORTING": "1",
                            "DISABLE_AUTO_COMPACT": "1", "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h",
                            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16000", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


def test_cli_rejecting_thinking_display_is_retried_without_it(cli):
    cli.script({"reject_thinking_display": True, **answer()})
    assert ask(cli.client()).choices[0].message.content == "Real answer."
    spawns = cli.spawns()
    assert len(spawns) == 2 and "--thinking-display" not in spawns[1]["args"]
    ask(cli.client())
    assert "--thinking-display" not in cli.spawns()[2]["args"]  # off for the process


@pytest.mark.parametrize("case", ["missing", "not-executable", "bad-cwd"])
def test_unlaunchable_cli_is_a_launch_error_that_falls_back(cli, tmp_path, case):
    not_executable = tmp_path / "claude-644"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(0o644)
    command, cwd = {"missing": (str(tmp_path / "gone" / "claude"), str(tmp_path)),
                    "not-executable": (str(not_executable), str(tmp_path)),
                    "bad-cwd": (cli.command, str(tmp_path / "no-such-dir"))}[case]
    client = ClaudeCodeClient(command=command, cwd=cwd)
    try:
        with pytest.raises(ClaudeCodeLaunchError) as info:
            ask(client)
    finally:
        client.close()
    assert not isinstance(info.value, OSError)  # the classifier would call that transport
    classified = classify_api_error(info.value, provider="claude-code", model=MODEL)
    assert classified.reason == FailoverReason.provider_unavailable
    assert classified.retryable is False and classified.should_fallback is True
    if case == "bad-cwd":
        assert "working directory" in str(info.value)


def test_each_cli_turn_is_logged(cli, caplog):
    cli.script({"events": [{"type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 10,
                            "error_status": 529}, *answer()["events"]]}, answer())
    client = cli.client()
    with caplog.at_level(logging.INFO, logger="agent.claude_code_session"):
        ask(client)
        ask(client, history(2))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Claude Code turn:")]
    assert len(lines) == 2
    assert "mode=cold" in lines[0] and "outcome=ok" in lines[0] and "api_retries=1" in lines[0]
    assert "mode=resume" in lines[1] and "out_tokens=2" in lines[1] and "ttft_ms=" in lines[1]


# ── Quiet work turns and liveness ────────────────────────────────────────


def tool_turn(narration):
    call = {"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}}
    return {"events": [*assistant(text(narration), call)[:-3],
                       {"type": "TOOL_CALL", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}}]}


def deltas(chunks, field):
    return "".join(getattr(c.choices[0].delta, field, None) or "" for c in chunks if c.choices)


def test_narration_before_a_tool_call_is_commentary_not_answer(cli):
    cli.script(tool_turn("Checking the logs."))
    chunks = list(ask(cli.client(), tools=TOOLS, stream=True))
    assert deltas(chunks, "content") == ""
    assert deltas(chunks, "commentary") == "Checking the logs."
    assert [c.choices[0].delta.tool_calls[0].id for c in chunks if c.choices and c.choices[0].delta.tool_calls] == ["toolu_1"]


def test_long_prose_before_a_call_still_streams_as_the_answer(cli):
    prose = "x" * (cce.STREAM_COMMIT_CHARS_WITH_TOOLS + 50)
    cli.script(tool_turn(prose))
    chunks = list(ask(cli.client(), tools=TOOLS, stream=True))
    assert deltas(chunks, "content") == prose and deltas(chunks, "commentary") == ""


def test_short_answer_without_a_call_is_delivered_as_content(cli):
    cli.script(answer("Done."))
    chunks = list(ask(cli.client(), tools=TOOLS, stream=True))
    assert deltas(chunks, "content") == "Done." and deltas(chunks, "commentary") == ""


def test_commentary_reaches_the_interim_rail_not_the_answer_stream(cli):
    """The generic consumer keeps commentary in the message content (history and the
    next tool-result prefix) but never fires the answer-stream callback for it."""
    from run_agent import AIAgent

    cli.script(tool_turn("Checking the logs."))
    agent = AIAgent(provider="claude-code", model="opus", base_url="acp://claude-code", api_key="claude-code",
                    command=cli.command, args=[], enabled_toolsets=[], skip_memory=True, skip_context_files=True,
                    quiet_mode=True, save_trajectories=False, max_iterations=2)
    streamed, interim = [], []
    agent.stream_delta_callback = streamed.append
    agent.interim_assistant_callback = lambda text, **_kw: interim.append(text)
    try:
        response = agent._interruptible_streaming_api_call(
            {"model": "opus", "messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})
    finally:
        agent.close()
    assert response.choices[0].message.content == "Checking the logs."
    assert "".join(t for t in streamed if t) == ""
    assert interim == ["Checking the logs."]


def test_liveness_ticks_flow_while_claude_writes_a_tool_call(cli, monkeypatch):
    monkeypatch.setattr(cce, "KEEPALIVE_SECONDS", 0.0)
    call = {"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}}
    cli.script({"events": [*assistant(call)[:-3],
                           {"type": "TOOL_CALL", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}}]})
    chunks = list(ask(cli.client(), tools=TOOLS, stream=True))
    empty = [c for c in chunks if c.choices and not any(
        getattr(c.choices[0].delta, f) for f in ("content", "reasoning_content", "tool_calls", "commentary"))
        and c.choices[0].finish_reason is None]
    assert empty  # all-empty deltas keep the consumer's stale-stream detector fresh


# ── Replay payload elision ───────────────────────────────────────────────


def call_row(call_id, content):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": f"/tmp/{call_id}", "content": content})}}]}


def test_full_replay_elides_long_arguments_of_older_calls_only():
    long = "y" * (ELIDE_ARG_CHARS + 1)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    for call_id in ("a", "b", "c"):
        messages += [call_row(call_id, long), {"role": "tool", "tool_call_id": call_id, "content": "ok"}]
    messages.append(call_row("d", "short"))

    def arguments(transcript):
        records = json.loads(transcript)["conversation_records"]
        return [json.loads(r["completed_historical_calls"][0]["function"]["arguments"])
                for r in records if r.get("completed_historical_calls")]

    elided = arguments(system_and_transcript(messages, elide_old_calls=True)[1])
    assert "more characters elided" in elided[0]["content"] and elided[0]["path"] == "/tmp/a"
    assert "more characters elided" in elided[1]["content"]
    # The two most recent tool-call replies keep their arguments exactly.
    assert elided[2]["content"] == long and elided[3]["content"] == "short"
    assert all(a["content"] in (long, "short") for a in arguments(system_and_transcript(messages)[1]))


# ── Conversation loop: Claude limits, fallback, the Claude error kept ────


class _BillingError(Exception):
    status_code = 402

    def __init__(self):
        super().__init__("Error code: 402 - Insufficient balance or no resource package")
        self.body = {"error": {"message": "Insufficient balance or no resource package"}}


def _limit():
    return usage_limit_error({"status": "rejected", "rateLimitType": "five_hour",
                              "resetsAt": int(time.time()) + 3600}, "You've hit your session limit · resets 3pm (UTC)")


@contextlib.contextmanager
def claude_agent(cli, monkeypatch, *, fallback=False, fallback_error=None):
    from run_agent import AIAgent

    agent = AIAgent(provider="claude-code", model=MODEL, base_url="acp://claude-code", api_key="claude-code",
                    command=cli.command, args=[], enabled_toolsets=[], skip_memory=True, skip_context_files=True,
                    quiet_mode=True, save_trajectories=False, max_iterations=4)
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    record = {"statuses": [], "reasons": [], "calls": 0}
    agent._fallback_chain = [{"provider": "zai", "model": "glm-5.3"}] if fallback else []
    agent._fallback_index = 0

    def call(*_args, **_kwargs):
        record["calls"] += 1
        raise fallback_error if agent.provider == "zai" else _limit()

    def activate(reason=None, **_kw):
        record["reasons"].append(reason)
        if agent._fallback_index >= len(agent._fallback_chain):
            return False
        agent._fallback_index += 1
        agent.provider, agent.model = "zai", "glm-5.3"
        return True

    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", call)
    monkeypatch.setattr(agent, "_interruptible_api_call", call)
    monkeypatch.setattr(agent, "_emit_status", record["statuses"].append)
    monkeypatch.setattr(agent, "_try_activate_fallback", activate)
    try:
        with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"):
            yield agent, record
    finally:
        agent.close()


def test_usage_limit_ends_the_turn_at_once_with_the_reset_time(cli, monkeypatch):
    with claude_agent(cli, monkeypatch) as (agent, record):
        result = agent.run_conversation("hello")
    assert record["calls"] == 1 and record["reasons"] == []
    assert result["failed"] is True and result["failure_reason"] == "rate_limit"
    assert result["final_response"].startswith("Claude 5-hour session limit reached — resets ")
    # Regression (958294bc31): the limit is the reply; a status line said it again.
    assert not any(s.startswith("⏳") for s in record["statuses"])


def test_usage_limit_is_announced_before_the_fallback_and_survives_its_failure(cli, monkeypatch):
    with claude_agent(cli, monkeypatch, fallback=True, fallback_error=_BillingError()) as (agent, record):
        result = agent.run_conversation("hello")
    assert record["reasons"][0] == FailoverReason.rate_limit
    assert [s for s in record["statuses"] if s.startswith("⏳")] == [
        s for s in record["statuses"] if s.startswith("⏳ Claude 5-hour session limit reached")] != []
    final = result["final_response"]
    assert final.startswith("Claude 5-hour session limit reached — resets ")
    assert "Fallback zai/glm-5.3 failed too: " in final and "Insufficient balance" in final


# ── Warm processes across user turns ────────────────────────────────────


def wait_for(predicate, timeout=5.0):
    until = time.monotonic() + timeout
    while not predicate() and time.monotonic() < until:
        time.sleep(0.02)
    return predicate()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_warm_process_serves_the_next_user_turn(cli, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")
    cli.script({**answer("First."), "next": [answer("Second.")["events"]]})
    client = cli.client()
    ask(client)
    pid = client._session._process.pid
    session_id = json.loads(checkpoints(cli)[0].read_text())["session_id"]
    with caplog.at_level(logging.INFO, logger="agent.claude_code_session"):
        second = ask(client, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "First."},
                              {"role": "user", "content": "more"}])
    assert second.choices[0].message.content == "Second."
    assert len(cli.spawns()) == 1 and client._session._process.pid == pid
    # Only the new record goes to the process; it already holds the conversation.
    assert cli.request(0, 1) == [{"role": "user", "text": "more"}]
    assert json.loads(checkpoints(cli)[0].read_text())["session_id"] == session_id
    assert any("mode=warm" in r.getMessage() and "outcome=ok" in r.getMessage() for r in caplog.records)


def test_warm_process_ends_after_its_keepalive(cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0.3")
    cli.script(answer("First."), answer("Second."))
    client = cli.client()
    ask(client)
    pid = client._session._process.pid
    assert wait_for(lambda: client._session._process is None and not alive(pid))
    ask(client, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "First."},
                 {"role": "user", "content": "more"}])
    assert len(cli.spawns()) == 2 and "--resume" in cli.spawns()[1]["args"]


def test_warm_process_is_not_reused_for_a_conversation_it_does_not_hold(cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")
    cli.script(answer("First."), answer("Other."))
    client = cli.client()
    ask(client)
    pid = client._session._process.pid
    # A rewind / edited history: the parked process holds a different transcript.
    assert ask(client, [{"role": "user", "content": "something else"}]).choices[0].message.content == "Other."
    assert len(cli.spawns()) == 2 and not alive(pid)


def test_warm_processes_are_bounded(cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")
    monkeypatch.setenv("HERMES_CLAUDE_CODE_MAX_WARM", "1")
    cli.script(answer())
    first = cli.client()
    ask(first)
    pid = first._session._process.pid
    token = set_native_transport_identity("another-conversation")
    try:
        second = cli.client()
        ask(second)
    finally:
        reset_native_transport_identity(token)
    assert second._session._process is not None
    assert wait_for(lambda: not alive(pid)) and first._session._process is None
