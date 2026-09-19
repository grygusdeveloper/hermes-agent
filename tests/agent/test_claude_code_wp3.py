"""Claude Code bridge: streaming, liveness, interrupts, CLI environment (WP3).

Most tests drive the real ``ClaudeCodeSession``/``ClaudeCodeClient`` against a
fake stream-json CLI that behaves like Claude Code 2.1.276 where it matters
here: with ``--include-partial-messages`` it streams ``message_start`` /
``content_block_*`` / ``message_delta`` / ``message_stop`` events, emits one
``assistant`` event (with a chain ``uuid`` and ``message.id``) per content
block, answers a stream-json ``control_request`` "interrupt" by ending the
turn with a "[Request interrupted by user]" user entry, persists its chain
for ``--resume``/``--resume-session-at``, and can script API retries,
output-limit continuations, system events and silence.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

import agent.claude_code_client as ccc
import agent.claude_code_session as ccs
from agent.claude_code_client import ClaudeCodeClient, _completion_parts
from agent.claude_code_session import (
    _STREAM_DIVERGED_SEPARATOR,
    _STREAM_RESTART_SEPARATOR,
    ClaudeCodeInterrupted,
    ClaudeCodeSession,
    ClaudeCodeSoftLimitNotice,
    _build_subprocess_env,
    _parse_claude_reply,
    _parse_stream_json_output,
    _parse_stream_json_usage,
    _Progress,
    _StreamGate,
    _TurnMonitor,
    _WARM_IDLE,
)

SID = "12345678-1234-1234-1234-123456789abc"

FAKE_CLI = textwrap.dedent(
    r"""
    import json, os, queue, signal, sys, threading, time, uuid
    log = os.environ["FAKE_CLAUDE_LOG"]
    store = os.environ["FAKE_CLAUDE_STORE"]
    script = os.environ.get("FAKE_CLAUDE_SCRIPT", "")
    argv = sys.argv[1:]
    partial = "--include-partial-messages" in argv
    MODEL = "claude-opus-5"
    ENV_KEYS = ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_AUTO_COMPACT",
                "CLAUDE_CODE_PROMPT_CACHE_TTL", "DISABLE_TELEMETRY"]

    def arg(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    def write_log(obj):
        with open(log, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    def on_term(signum, frame):
        write_log({"event": "sigterm"})
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)

    def next_reply():
        if not script or not os.path.exists(script):
            return {}
        counter = script + ".n"
        n = int(open(counter).read()) if os.path.exists(counter) else 0
        replies = json.load(open(script))
        with open(counter, "w") as fh:
            fh.write(str(n + 1))
        return replies[n] if n < len(replies) else {}

    write_log({"event": "spawn", "argv": argv, "pid": os.getpid(),
               "env": {k: os.environ.get(k) for k in ENV_KEYS}})
    if "--thinking-display" in argv and os.environ.get("FAKE_REJECT_THINKING_DISPLAY"):
        sys.stderr.write("error: unknown option '--thinking-display'\n")
        sys.exit(1)
    sid = arg("--resume") or arg("--session-id")
    resume_at = arg("--resume-session-at")
    persist = "--no-session-persistence" not in argv
    path = os.path.join(store, sid + ".json")
    chain = []
    if "--resume" in argv:
        if not os.path.exists(path):
            sys.stderr.write("No conversation found with session ID: " + sid + "\n")
            sys.exit(1)
        chain = json.load(open(path))
        if resume_at:
            idx = next((i for i, e in enumerate(chain) if e["uuid"] == resume_at), None)
            if idx is None:
                sys.stderr.write("No message found with message.uuid of: " + resume_at + "\n")
                sys.exit(1)
            chain = chain[: idx + 1]

    def save():
        if persist:
            with open(path, "w") as fh:
                json.dump(chain, fh)

    out_lock = threading.Lock()

    def out(obj):
        with out_lock:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()

    def stream(event):
        if partial:
            out({"type": "stream_event", "event": event, "session_id": sid})

    inbox = queue.Queue()

    def reader():
        for line in sys.stdin:
            inbox.put(line)
        inbox.put(None)

    threading.Thread(target=reader, daemon=True).start()
    deferred = []

    def interrupt_requested():
        while True:
            try:
                line = inbox.get_nowait()
            except queue.Empty:
                return False
            if line is None:
                deferred.append(None)
                return False
            msg = json.loads(line)
            if msg.get("type") == "control_request":
                out({"type": "control_response", "response": {
                    "subtype": "success", "request_id": msg["request_id"],
                    "response": {"still_queued": []}}})
                write_log({"event": "interrupt"})
                return True
            deferred.append(line)

    class Interrupted(Exception):
        pass

    def entry(text):
        item = {"uuid": str(uuid.uuid4()), "text": text}
        chain.append(item)
        return item["uuid"]

    def message(text, reply, *, stop_reason="end_turn", thinking=None, abandon=None):
        msg_id = "msg_" + uuid.uuid4().hex[:16]
        stream({"type": "message_start", "message": {"id": msg_id, "model": MODEL}})
        if thinking:
            stream({"type": "content_block_start", "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""}})
            for k in range(reply.get("thinking_tokens", 0)):
                out({"type": "system", "subtype": "thinking_tokens",
                     "estimated_tokens": 10 * (k + 1), "estimated_tokens_delta": 10})
                time.sleep(reply.get("delay", 0))
            stream({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": thinking}})
            stream({"type": "content_block_stop", "index": 0})
            out({"type": "assistant", "uuid": entry("(thinking)"), "session_id": sid,
                 "message": {"id": msg_id, "model": MODEL,
                             "content": [{"type": "thinking", "thinking": thinking}]}})
        stream({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": ""}})
        step = int(reply.get("chunk", 7))
        shown = text if abandon is None else text[:abandon]
        for i in range(0, len(shown), step):
            stream({"type": "content_block_delta", "index": 1,
                    "delta": {"type": "text_delta", "text": shown[i:i + step]}})
            time.sleep(reply.get("delay", 0))
            if not reply.get("ignore_interrupt") and interrupt_requested():
                out({"type": "assistant", "uuid": entry("A: " + shown[:i + step]),
                     "session_id": sid, "message": {"id": msg_id, "model": MODEL,
                     "content": [{"type": "text", "text": shown[:i + step]}]}})
                marker = entry("[Request interrupted by user]")
                out({"type": "user", "uuid": marker, "session_id": sid, "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "[Request interrupted by user]"}]}})
                save()
                raise Interrupted()
        stream({"type": "content_block_stop", "index": 1})
        if abandon is not None:
            # Claude Code's connection-retry branch: synthetic block/message
            # stop without a message_delta, then an api_retry notice.
            if reply.get("abandon_assistant", True):
                out({"type": "assistant", "uuid": str(uuid.uuid4()), "session_id": sid,
                     "message": {"id": msg_id, "model": MODEL,
                                 "content": [{"type": "text", "text": shown}]}})
            stream({"type": "message_stop"})
            out({"type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 10,
                 "retry_delay_ms": 500, "error_status": None, "error": "ECONNRESET"})
            return None
        uid = entry("A: " + text)
        out({"type": "assistant", "uuid": uid, "session_id": sid,
             "message": {"id": msg_id, "model": MODEL,
                         "content": [{"type": "text", "text": text}]}})
        stream({"type": "message_delta", "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 5}})
        stream({"type": "message_stop"})
        return uid

    def result(text, **extra):
        body = {"type": "result", "subtype": "success", "is_error": False, "result": text,
                "session_id": sid, "stop_reason": "end_turn", "terminal_reason": "completed",
                "duration_api_ms": 12, "ttft_ms": 7,
                "usage": {"input_tokens": 3, "output_tokens": 5}}
        body.update(extra)
        out(body)

    while True:
        line = deferred.pop(0) if deferred else inbox.get()
        if line is None:
            break
        msg = json.loads(line)
        if msg.get("type") == "control_request":
            out({"type": "control_response", "response": {
                "subtype": "success", "request_id": msg["request_id"],
                "response": {"still_queued": []}}})
            continue
        content = msg["message"]["content"]
        text_in = content if isinstance(content, str) else json.dumps(content)
        entry(text_in)
        save()
        write_log({"event": "turn", "content": content, "context": [e["text"] for e in chain]})
        reply = next_reply()
        time.sleep(float(reply.get("sleep", 0)))
        out({"type": "system", "subtype": "init", "session_id": sid})
        for event in reply.get("system", []):
            out(dict(event, type="system", session_id=sid))
        if reply.get("silent"):
            time.sleep(float(reply["silent"]))
        text = reply.get("text", "Fake reply.")
        try:
            if reply.get("abandon") is not None:
                message(reply.get("abandoned", text), reply, abandon=int(reply["abandon"]))
            if reply.get("split") is not None:
                cut = int(reply["split"])
                message(text[:cut], reply, stop_reason="max_tokens")
                message(text[cut:], reply, thinking=reply.get("thinking"))
                last = text[cut:]
            else:
                message(text, reply, thinking=reply.get("thinking"))
                last = text
        except Interrupted:
            out({"type": "result", "subtype": "error_during_execution", "is_error": True,
                 "session_id": sid, "stop_reason": None, "terminal_reason": "aborted_streaming",
                 "result": "", "usage": {"input_tokens": 0, "output_tokens": 0}})
            continue
        save()
        result(last)
    """
)


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI)
    launcher = tmp_path / "claude"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    store = tmp_path / "store"
    store.mkdir()
    log = tmp_path / "calls.jsonl"
    replies = tmp_path / "replies.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_STORE", str(store))
    monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", str(replies))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")
    monkeypatch.delenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_GRACEFUL_INTERRUPT", raising=False)
    monkeypatch.setattr(ccs, "_thinking_display_supported", True)

    def events(kind=None):
        if not log.exists():
            return []
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return [row for row in rows if kind is None or row["event"] == kind]

    def script_replies(*items):
        replies.write_text(json.dumps(list(items)))
        counter = replies.with_name(replies.name + ".n")
        if counter.exists():
            counter.unlink()

    yield SimpleNamespace(command=str(launcher), events=events, script=script_replies, store=store)
    for warm in list(_WARM_IDLE.values()):
        warm.close()


def _env(**extra):
    keys = ("PATH", "FAKE_CLAUDE_LOG", "FAKE_CLAUDE_STORE", "FAKE_CLAUDE_SCRIPT",
            "FAKE_REJECT_THINKING_DISPLAY")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env.update(extra)
    return env


def _run(session, cli, messages, *, state_key="root|main", tools=True, **kwargs):
    return session.run(
        "FULL PROMPT",
        messages=messages,
        model="opus",
        tools_digest="digest" if tools else "",
        has_tools=tools,
        timeout_seconds=30,
        cwd="/tmp",
        env=kwargs.pop("env", None) or _env(),
        state_key=state_key,
        command=cli.command,
        system_prompt="SYSTEM",
        keepalive=kwargs.pop("keepalive", True),
        **kwargs,
    )


def _call(name="terminal", call_id="c1", arguments=None):
    return "<tool_call>" + json.dumps(
        {"id": call_id, "name": name, "arguments": arguments or {"command": "ls"}}
    ) + "</tool_call>"


def _wait_for(predicate, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _feed(gate, text, step=7):
    for index in range(0, len(text), step):
        gate.feed(text[index:index + step])


# ---------------------------------------------------------------------------
# thinking-display-omitted
# ---------------------------------------------------------------------------


def _argv(session, **kwargs):
    return session._build_argv(
        "claude", session_id=None, model="opus", system_prompt=None,
        stream_partials=True, **kwargs,
    )


def test_thinking_display_is_sent_whatever_the_effort(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", raising=False)
    monkeypatch.setattr(ccs, "_thinking_display_supported", True)
    session = ClaudeCodeSession()
    for effort in (None, "high"):
        argv = _argv(session, effort=effort)
        assert argv[argv.index("--thinking-display") + 1] == "summarized"
    monkeypatch.setenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", "omitted")
    argv = _argv(session, effort=None)
    assert argv[argv.index("--thinking-display") + 1] == "omitted"
    for value in ("off", "full", ""):
        monkeypatch.setenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", value)
        # "" means the default; anything outside the CLI's choices sends
        # nothing (the CLI would exit 1 on it).
        assert ("--thinking-display" in _argv(session, effort=None)) is (value == "")


def test_thinking_display_is_part_of_the_warm_identity(fake_cli, monkeypatch):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "again"}]
    _run(session, fake_cli, history)
    assert len(fake_cli.events("spawn")) == 1  # warm reuse
    monkeypatch.setenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", "omitted")
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "3"}]
    _run(session, fake_cli, history)
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2
    assert spawns[1]["argv"][spawns[1]["argv"].index("--thinking-display") + 1] == "omitted"
    assert "--resume" in spawns[1]["argv"]
    session.shutdown()


def test_cli_rejecting_thinking_display_is_retried_without_it(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_REJECT_THINKING_DISPLAY", "1")
    session = ClaudeCodeSession()
    response, _reasoning = _run(session, fake_cli, [{"role": "user", "content": "hi"}])
    assert response == "Fake reply."
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2
    assert "--thinking-display" in spawns[0]["argv"]
    assert "--thinking-display" not in spawns[1]["argv"]
    assert ccs._thinking_display_supported is False
    session.shutdown()


def test_invalid_thinking_display_error_turns_the_flag_off(monkeypatch):
    monkeypatch.setattr(ccs, "_thinking_display_supported", True)
    session = ClaudeCodeSession()
    with pytest.raises(ccs._CliFlagRejected):
        session._raise_process_failure(
            1, "", "error: option '--thinking-display <display>' argument 'full' is "
            "invalid. Allowed choices are summarized, omitted.", None,
        )
    assert ccs._thinking_display() is None


def test_summarized_thinking_streams_as_reasoning(fake_cli):
    fake_cli.script({"text": "Done.", "thinking": "Weighing the options."})
    session = ClaudeCodeSession()
    thoughts = []
    response, reasoning = _run(
        session, fake_cli, [{"role": "user", "content": "hi"}],
        on_reasoning_chunk=thoughts.append, on_text_chunk=lambda _t: None,
    )
    assert response == "Done."
    assert thoughts == ["Weighing the options."]
    assert reasoning == "Weighing the options."
    session.shutdown()


# ---------------------------------------------------------------------------
# cli-env-nonessential-traffic + claude-autocompact-unmanaged
# ---------------------------------------------------------------------------


def test_child_env_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("DISABLE_AUTO_COMPACT", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", raising=False)
    env = _build_subprocess_env({"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret"})
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTO_COMPACT"] == "1"
    assert env["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "1h"
    assert "ANTHROPIC_API_KEY" not in env
    # An operator's explicit value wins (base env or Hermes's environment).
    assert _build_subprocess_env({"DISABLE_AUTO_COMPACT": "0"})["DISABLE_AUTO_COMPACT"] == "0"
    monkeypatch.setenv("CLAUDE_CODE_PROMPT_CACHE_TTL", "5m")
    assert _build_subprocess_env({})["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "5m"


def test_spawned_cli_gets_the_env_defaults(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, [{"role": "user", "content": "hi"}])
    env = fake_cli.events("spawn")[0]["env"]
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTO_COMPACT"] == "1"
    session.shutdown()


# ---------------------------------------------------------------------------
# abort-blocks-before-signal
# ---------------------------------------------------------------------------


def _execute(session, cli, **kwargs):
    return session._execute(
        "hello",
        session_id=None,
        model="opus",
        effort=None,
        timeout_seconds=kwargs.pop("timeout_seconds", 30),
        cwd="/tmp",
        env=_env(),
        command=cli.command,
        keepalive=True,
        **kwargs,
    )


def test_timeout_kills_a_silent_cli(fake_cli):
    """Regression: the watchdog's abort() closed stdout first, which blocked
    on the reader's lock until the CLI printed again: a silent CLI was never
    killed (the call returned after the CLI's own 30 s)."""

    fake_cli.script({"silent": 30})
    session = ClaudeCodeSession()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        _execute(session, fake_cli, timeout_seconds=0.3)
    assert time.monotonic() - started < 8.0
    assert _wait_for(lambda: fake_cli.events("sigterm"), timeout=5)


def test_abort_returns_at_once_and_kills_a_silent_cli(fake_cli):
    fake_cli.script({"silent": 30})
    session = ClaudeCodeSession()
    box = {}

    def target():
        try:
            _execute(session, fake_cli)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc
        box["done"] = time.monotonic()

    thread = threading.Thread(target=target)
    thread.start()
    assert _wait_for(lambda: fake_cli.events("turn"))
    time.sleep(0.2)  # the reader is blocked in readline() now
    started = time.monotonic()
    session.abort()
    assert time.monotonic() - started < 0.2
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "aborted" in str(box["exc"])
    assert box["done"] - started < 3.0
    assert _wait_for(lambda: fake_cli.events("sigterm"), timeout=5)


# ---------------------------------------------------------------------------
# interrupt-lost-between-attempts
# ---------------------------------------------------------------------------


def _fake_run(session, **kwargs):
    return session.run(
        "FULL",
        messages=[{"role": "user", "content": "hi"}],
        model="opus",
        tools_digest="digest",
        cwd="/tmp",
        env={},
        state_key=None,
        **kwargs,
    )


def test_abort_during_soft_limit_backoff_stops_the_run(monkeypatch):
    session = ClaudeCodeSession()
    calls = []

    def fake_execute(prompt, **kwargs):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise ClaudeCodeSoftLimitNotice("rate_limit 429")
        return "late answer", "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    box = {}

    def target():
        try:
            box["result"] = _fake_run(session)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc
        box["done"] = time.monotonic()

    thread = threading.Thread(target=target)
    thread.start()
    assert _wait_for(lambda: calls)
    time.sleep(0.2)  # inside the 2 s backoff
    aborted_at = time.monotonic()
    session.abort()
    thread.join(timeout=5)
    assert isinstance(box.get("exc"), ClaudeCodeInterrupted)
    assert box["done"] - aborted_at < 0.5
    assert len(calls) == 1  # the second attempt never started


def test_abort_cancels_a_run_queued_behind_another(monkeypatch):
    session = ClaudeCodeSession()
    release = threading.Event()
    started = threading.Event()
    calls = []

    def fake_execute(prompt, **kwargs):
        calls.append(prompt)
        started.set()
        release.wait(5)
        return "answer", "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    results = {}

    def target(name):
        try:
            results[name] = _fake_run(session)
        except BaseException as exc:  # noqa: BLE001
            results[name] = exc

    first = threading.Thread(target=target, args=("first",))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=target, args=("second",))
    second.start()
    time.sleep(0.2)  # queued on the session lock
    session.abort()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert isinstance(results["second"], ClaudeCodeInterrupted)
    assert len(calls) == 1


def test_abort_run_only_cancels_its_own_run(monkeypatch):
    session = ClaudeCodeSession()
    calls = []

    def fake_execute(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise ClaudeCodeSoftLimitNotice("rate_limit 429")
        return "answer", "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    mine, other = threading.Event(), threading.Event()
    box = {}

    def target():
        try:
            box["result"] = _fake_run(session, cancel_event=mine)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    assert _wait_for(lambda: calls)
    session.abort_run(other)  # someone else's run: no effect here
    thread.join(timeout=5)
    assert box["result"][0] == "answer"
    assert not mine.is_set()


# ---------------------------------------------------------------------------
# Graceful interrupt of a warm process (control_request "interrupt")
# ---------------------------------------------------------------------------

LONG = "counting " * 200


def _interrupt_second_turn(session, cli, history):
    """Run ``history`` in a thread and abort once text streams; return the error."""

    box = {}
    streamed = threading.Event()

    def target():
        try:
            _run(session, cli, history, on_text_chunk=lambda _t: streamed.set())
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc
        box["done"] = time.monotonic()

    thread = threading.Thread(target=target)
    thread.start()
    assert _wait_for(lambda: len(cli.events("turn")) >= 2)
    time.sleep(0.3)
    aborted_at = time.monotonic()
    session.abort()
    thread.join(timeout=10)
    assert not thread.is_alive()
    return box["exc"], box["done"] - aborted_at


def test_interrupt_keeps_the_warm_process_and_resumes_after_it(fake_cli):
    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5}, {"text": "Next answer."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count please"}]
    exc, elapsed = _interrupt_second_turn(session, fake_cli, history)
    assert isinstance(exc, ClaudeCodeInterrupted)
    assert elapsed < 1.5
    assert fake_cli.events("interrupt") and not fake_cli.events("sigterm")
    warm = session._warm
    assert warm is not None and warm.alive() and warm.tip == session._interrupted_turn.marker

    # Hermes keeps the interrupted request, adds its scaffold and the new message.
    history += [
        {"role": "assistant", "content": "[This response was interrupted by a user correction.]"},
        {"role": "user", "content": "stop, do this instead"},
    ]
    response, _ = _run(session, fake_cli, history)
    assert response == "Next answer."
    assert len(fake_cli.events("spawn")) == 1  # same process throughout
    turn = fake_cli.events("turn")[-1]
    assert "stop, do this instead" in turn["content"]
    assert "count please" not in turn["content"]  # never sent twice
    assert turn["context"][-2] == "[Request interrupted by user]"
    assert session._interrupted_turn is None
    session.shutdown()


def test_interrupted_request_resumes_cold_at_the_marker(fake_cli):
    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5}, {"text": "Next."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count"}]
    _interrupt_second_turn(session, fake_cli, history)
    marker = session._interrupted_turn.marker
    session._discard_warm()  # e.g. evicted while idle
    history += [{"role": "user", "content": "new question"}]
    assert _run(session, fake_cli, history)[0] == "Next."
    spawn = fake_cli.events("spawn")[-1]["argv"]
    assert spawn[spawn.index("--resume-session-at") + 1] == marker
    turn = fake_cli.events("turn")[-1]
    assert "new question" in turn["content"] and "count" not in turn["content"]
    session.shutdown()


def test_fallback_answer_after_an_interrupted_request_is_replayed(fake_cli):
    """Regression (H1): an answer another provider gave after the interrupted
    request is not in the Claude session; continuing at the marker with only
    the user messages would drop it."""

    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5}, {"text": "Next."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count"}]
    _interrupt_second_turn(session, fake_cli, history)
    history += [
        {"role": "assistant", "content": ""},  # hidden placeholder: nothing to miss
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "FALLBACK ANSWER: PINEAPPLE"},
        {"role": "user", "content": "and now?"},
    ]
    assert _run(session, fake_cli, history)[0] == "Next."
    assert "--session-id" in fake_cli.events("spawn")[-1]["argv"]
    assert fake_cli.events("turn")[-1]["content"] == "FULL PROMPT"
    session.shutdown()


def test_unrelated_next_request_ignores_the_interrupted_one(fake_cli):
    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5}, {"text": "Other."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    published = session._checkpoints[-1][1]
    interrupted = history + [
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "count"},
    ]
    _interrupt_second_turn(session, fake_cli, interrupted)
    # /undo-like: the interrupted request is gone from Hermes's history.
    rewritten = history + [
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "something else"},
    ]
    assert _run(session, fake_cli, rewritten)[0] == "Other."
    spawn = fake_cli.events("spawn")[-1]["argv"]
    assert spawn[spawn.index("--resume-session-at") + 1] == published
    session.shutdown()


def test_cli_ignoring_the_interrupt_is_killed(fake_cli, monkeypatch):
    monkeypatch.setattr(ccs, "_INTERRUPT_GRACE_SECONDS", 0.3)
    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5, "ignore_interrupt": True})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count"}]
    exc, elapsed = _interrupt_second_turn(session, fake_cli, history)
    assert "aborted" in str(exc)
    assert elapsed < 4.0
    assert _wait_for(lambda: fake_cli.events("sigterm"), timeout=5)
    assert session._interrupted_turn is None
    session.shutdown()


def test_graceful_interrupts_can_be_disabled(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_GRACEFUL_INTERRUPT", "0")
    fake_cli.script({"text": "Hello."}, {"text": LONG, "delay": 0.02, "chunk": 5})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count"}]
    exc, _elapsed = _interrupt_second_turn(session, fake_cli, history)
    assert "aborted" in str(exc)
    assert not fake_cli.events("interrupt")
    assert _wait_for(lambda: fake_cli.events("sigterm"), timeout=5)
    session.shutdown()


def test_abandoned_stream_cancels_its_request(fake_cli):
    fake_cli.script({"text": LONG, "delay": 0.02, "chunk": 5})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    stream = client._create_chat_completion(
        model="opus", messages=[{"role": "user", "content": "count"}], stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            break
    started = time.monotonic()
    stream.close()
    assert time.monotonic() - started < 3.0
    # A tool-less, stateless request cannot be interrupted gracefully: killed.
    assert _wait_for(lambda: fake_cli.events("sigterm"), timeout=5)
    client.close()


# ---------------------------------------------------------------------------
# stream-gate-diverges-on-cli-retry
# ---------------------------------------------------------------------------


def test_gate_restart_before_anything_was_shown_is_invisible():
    out = []
    gate = _StreamGate(out.append, commit_chars=40)
    gate.feed("The first attempt of the")
    gate.restart()
    final = "A complete answer that is definitely long enough to be committed."
    _feed(gate, final)
    assert gate.finish(final) == final
    assert "".join(out) == final


def test_gate_restart_after_text_was_shown_keeps_it_and_appends_the_answer():
    out = []
    gate = _StreamGate(out.append, commit_chars=10)
    partial = "Here is the report so far, section one of"
    _feed(gate, partial)
    shown = "".join(out)
    assert shown
    gate.restart()
    gate.restart()  # idempotent
    final = "Here is the report.\n\nMEDIA:/tmp/report.pdf"
    _feed(gate, final)
    answer = gate.finish(final)
    assert answer == shown + _STREAM_RESTART_SEPARATOR + final
    assert "".join(out) == answer.strip()
    assert answer.endswith("MEDIA:/tmp/report.pdf")


def test_gate_restart_closes_an_open_fence():
    out = []
    gate = _StreamGate(out.append, commit_chars=10)
    _feed(gate, "Run this:\n\n```bash\necho one\necho two\necho three and more")
    gate.restart()
    final = "Run this:\n\n" + _call()
    _feed(gate, final)
    answer = gate.finish(final)
    assert "\n```" + _STREAM_RESTART_SEPARATOR in answer
    # The call after the separator is outside the closed fence.
    assert len(_parse_claude_reply(answer).executable_calls) == 1
    assert "".join(out) == _parse_claude_reply(answer).cleaned


def test_gate_finish_keeps_shown_text_when_the_answer_diverges():
    out = []
    gate = _StreamGate(out.append, commit_chars=5)
    _feed(gate, "Something that was streamed first and then")
    shown = "".join(out)
    answer = gate.finish("A different final answer. MEDIA:/tmp/x.png")
    assert answer == shown + _STREAM_DIVERGED_SEPARATOR + "A different final answer. MEDIA:/tmp/x.png"
    assert "".join(out) == answer.strip()


def _stream(client, **kwargs):
    content, calls, finish, empty = [], [], [], 0
    for chunk in client._create_chat_completion(stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content.append(delta.content)
        if delta.tool_calls:
            calls.extend(delta.tool_calls)
        if not (delta.content or delta.tool_calls or delta.reasoning_content) and not chunk.choices[0].finish_reason:
            empty += 1
        if chunk.choices[0].finish_reason:
            finish.append(chunk.choices[0].finish_reason)
    return "".join(content), calls, finish, empty


TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]


@pytest.mark.parametrize("keepalive", ["0", "30"])
@pytest.mark.parametrize("tools", [True, False])
def test_cli_retry_before_commit_streams_only_the_real_answer(fake_cli, monkeypatch, keepalive, tools):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", keepalive)
    answer = "The deployment finished. " * 20 + "\n\nMEDIA:/tmp/hermes/report.pdf"
    fake_cli.script({"text": answer, "abandoned": "The deploy", "abandon": 10})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, _calls, finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "status?"}],
        tools=TOOLS if tools else None,
    )
    assert content == answer.strip()
    assert finish == ["stop"]
    client.close()


@pytest.mark.parametrize("tools", [True, False])
def test_cli_retry_after_commit_keeps_both_and_the_media_tag(fake_cli, tools):
    from gateway.platforms.base import BasePlatformAdapter

    answer = "Fresh full answer. " * 25 + "\n\nMEDIA:/tmp/hermes/report.pdf"
    abandoned = "An earlier attempt that streamed for a while. " * 12
    fake_cli.script({"text": answer, "abandoned": abandoned, "abandon": 400})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, _calls, _finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "status?"}],
        tools=TOOLS if tools else None,
    )
    assert content.endswith(answer.strip())
    assert _STREAM_RESTART_SEPARATOR.strip() in content
    media, _cleaned = BasePlatformAdapter.extract_media(content)
    assert [path for path, _voice in media] == ["/tmp/hermes/report.pdf"]
    client.close()


def test_cli_retry_after_a_previewed_call_reopens_the_gate(fake_cli):
    call = _call(arguments={"command": "uptime"})
    fake_cli.script({"text": "Checking uptime now.\n\n" + call,
                     "abandoned": "Checking uptime.\n\n<tool_call>{\"id\": \"c1\", \"na", "abandon": 40})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, calls, finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "uptime?"}], tools=TOOLS,
    )
    assert content == "Checking uptime." + _STREAM_RESTART_SEPARATOR.rstrip() + "\n\nChecking uptime now."
    assert [c.function.name for c in calls] == ["terminal"] and finish == ["tool_calls"]
    client.close()


def test_warm_process_from_a_non_streaming_call_streams_live(fake_cli):
    """Regression (real-CLI probe): the first process was spawned without
    --include-partial-messages, so a streaming call reusing it got no deltas
    until the result arrived."""

    answer = "A longer answer that streams in pieces as it is written. " * 4
    fake_cli.script({"text": "First."}, {"text": answer, "chunk": 9})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history, tools=False)
    history += [{"role": "assistant", "content": "First."}, {"role": "user", "content": "more"}]
    chunks = []
    assert _run(session, fake_cli, history, tools=False, on_text_chunk=chunks.append)[0] == answer
    assert len(fake_cli.events("spawn")) == 1
    assert "--include-partial-messages" in fake_cli.events("spawn")[0]["argv"]
    assert len(chunks) > 3 and "".join(chunks) == answer.strip()
    session.shutdown()


def test_monitor_detects_restarts_but_not_output_limit_continuations():
    seen = []
    monitor = _TurnMonitor(lambda kind, text: seen.append(kind), _Progress(),
                           mode="warm", pid=1, started=time.monotonic(), payload_chars=0)

    def stream(event):
        monitor.observe({"type": "stream_event", "event": event})

    def message(stop_reason):
        stream({"type": "message_start", "message": {"id": f"m{len(seen)}"}})
        stream({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}})
        if stop_reason:
            stream({"type": "message_delta", "delta": {"stop_reason": stop_reason}})
        stream({"type": "message_stop"})

    message(None)  # abandoned: nothing yet (the turn could end here)
    assert "restart" not in seen
    message("end_turn")  # the re-streamed message
    assert seen.count("restart") == 1
    message("max_tokens")  # a second message after a normal one also replaces it
    assert seen.count("restart") == 2
    message("end_turn")  # continuation after the output limit: joins
    assert seen.count("restart") == 2


def test_monitor_names_a_refusal_restart():
    seen = []
    monitor = _TurnMonitor(lambda kind, text: seen.append((kind, text)), _Progress(),
                           mode="warm", pid=1, started=time.monotonic(), payload_chars=0)

    def stream(event):
        monitor.observe({"type": "stream_event", "event": event})

    def message(stop_reason, *, refusal_event=False):
        stream({"type": "message_start", "message": {"id": f"m{len(seen)}"}})
        stream({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}})
        if refusal_event:
            monitor.observe({"type": "system", "subtype": "model_refusal_fallback",
                             "original_model": "claude-opus-5",
                             "fallback_model": "claude-opus-4-8"})
        if stop_reason:
            stream({"type": "message_delta", "delta": {"stop_reason": stop_reason}})
        stream({"type": "message_stop"})

    message("refusal")
    message("end_turn")  # the fallback model answers again: "refusal"
    message(None, refusal_event=True)  # replaces a normal message: ""
    message("end_turn")  # the CLI event alone names the refusal: "refusal"
    message(None)  # ""
    message("end_turn")  # after a dropped connection: ""
    assert [text for kind, text in seen if kind == "restart"] == [
        "refusal", "", "refusal", "", ""
    ]


def test_gate_restart_after_a_refusal_says_so():
    out = []
    gate = _StreamGate(out.append, commit_chars=5)
    _feed(gate, "A reply the safeguards stopped halfway through")
    shown = "".join(out)
    gate.restart("refusal")
    final = "The fallback model's answer."
    _feed(gate, final)
    answer = gate.finish(final)
    assert answer == shown + ccs._STREAM_REFUSAL_RESTART_SEPARATOR + final
    assert "Connection to Claude dropped" not in answer


@pytest.mark.parametrize("tools", [True, False])
def test_without_a_display_an_abandoned_attempt_never_reaches_the_answer(fake_cli, tools):
    """Regression (f2): with streaming off nothing was shown live, yet the
    abandoned attempt, a separator and the full answer were all delivered."""

    answer = "Fresh full answer. " * 25 + "\n\nMEDIA:/tmp/hermes/report.pdf"
    abandoned = "An earlier attempt that streamed for a while. " * 12
    fake_cli.script({"text": answer, "abandoned": abandoned, "abandon": 400})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, _calls, finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "status?"}],
        tools=TOOLS if tools else None, live_display=False,
    )
    assert content == answer.strip()
    assert finish == ["stop"]
    client.close()


def test_agent_tells_the_bridge_whether_the_reply_is_displayed():
    from agent.chat_completion_helpers import _add_claude_code_reasoning

    agent = SimpleNamespace(provider="claude-code", base_url="acp://claude-code",
                            reasoning_config=None, _has_stream_consumers=lambda: False)
    kwargs = {}
    _add_claude_code_reasoning(agent, kwargs)
    assert kwargs == {"live_display": False}
    agent._has_stream_consumers = lambda: True
    _add_claude_code_reasoning(agent, kwargs)
    assert kwargs == {"live_display": True}


def test_client_runs_without_a_text_callback_when_nothing_shows_it(monkeypatch):
    client = ClaudeCodeClient(cwd="/tmp")
    seen = {}

    def fake_run(prompt, *, on_text_chunk=None, on_activity=None, **kwargs):
        seen["on_text_chunk"] = on_text_chunk
        seen["on_activity"] = on_activity
        return "Answer.", ""

    monkeypatch.setattr(client._claude_session, "run", fake_run)
    content, _calls, finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "q"}], live_display=False
    )
    assert seen["on_text_chunk"] is None and seen["on_activity"] is not None
    assert content == "Answer." and finish == ["stop"]
    list(client._create_chat_completion(
        model="opus", messages=[{"role": "user", "content": "q"}], stream=True
    ))
    assert seen["on_text_chunk"] is not None  # default: a live display
    client.close()


# ---------------------------------------------------------------------------
# stream-gate-hides-progress-sentence / progress-sentence-held-until-toolcall-done
# ---------------------------------------------------------------------------


def test_progress_sentence_is_shown_as_soon_as_the_call_starts():
    out = []
    gate = _StreamGate(out.append)
    _feed(gate, "I'll check the servers.\n\n<tool_call>", step=3)
    assert out == []  # "{" not seen yet: could still be a mention
    gate.feed('{"id": "c1", "name": "terminal", "arguments": {"comm')
    assert "".join(out) == "I'll check the servers."
    assert gate.previewed and not gate.committed
    _feed(gate, 'and": "ls"}}</tool_call>')
    reply = "I'll check the servers.\n\n" + _call()
    assert gate.finish(reply) == reply
    assert "".join(out) == _parse_claude_reply(reply).cleaned


def test_mentions_are_not_previewed():
    for text in ("Use the `<tool_call>` tag", "Hermes closes calls with </tool_call> here and"):
        out = []
        gate = _StreamGate(out.append)
        _feed(gate, text + " " * 70 + "more words")
        assert out == [] and not gate.previewed


def test_repair_after_a_previewed_sentence_does_not_repeat_it(monkeypatch):
    session = ClaudeCodeSession()
    broken = "Checking the disk.\n\n<tool_call>{\"id\": \"c1\", \"name\": \"terminal\", \"arguments\": {\"command\": \"df"
    replies = iter([broken, _call(arguments={"command": "df -h"})])
    prompts = []

    def fake_execute(prompt, *, on_event=None, **kwargs):
        prompts.append(prompt)
        text = next(replies)
        for index in range(0, len(text), 5):
            on_event("text", text[index:index + 5])
        return text, "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    chunks = []
    response, _r, _sid = session._execute_with_soft_limit_retry(
        "prompt", session_id=SID, model="opus", effort=None, timeout_seconds=5,
        cwd="/tmp", env={}, on_text_chunk=chunks.append, had_tools=True,
    )
    assert "".join(chunks) == "Checking the disk."
    reply = _parse_claude_reply(response)
    assert reply.cleaned == "Checking the disk."
    assert len(reply.executable_calls) == 1
    assert "continue the answer after the text already shown" in prompts[1]


def test_retry_after_a_previewed_sentence_keeps_it_once(monkeypatch):
    """A 429 after the progress sentence was shown: the retry's reply follows
    the shown sentence instead of being glued to it or replacing it."""

    monkeypatch.setattr(ccs.time, "sleep", lambda _seconds: None)
    session = ClaudeCodeSession()
    opener = "Checking the disk.\n\n<tool_call>{\"id\": \"c1\", \"na"
    final = "Checking the disk now.\n\n" + _call(arguments={"command": "df -h"})
    calls = []

    def fake_execute(prompt, *, on_event=None, **kwargs):
        calls.append(prompt)
        text = opener if len(calls) == 1 else final
        for index in range(0, len(text), 4):
            on_event("text", text[index:index + 4])
        if len(calls) == 1:
            raise ClaudeCodeSoftLimitNotice("rate_limit 429")
        return text, "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    chunks = []
    response, _r, _sid = session._execute_with_soft_limit_retry(
        "prompt", session_id=SID, model="opus", effort=None, timeout_seconds=5,
        cwd="/tmp", env={}, on_text_chunk=chunks.append, had_tools=True,
    )
    joined = "".join(chunks)
    assert joined == "Checking the disk.\n\nChecking the disk now."
    reply = _parse_claude_reply(response)
    assert reply.cleaned == joined and len(reply.executable_calls) == 1


# ---------------------------------------------------------------------------
# no-liveness-during-tool-json + runtime-activity-and-cli-events-unbridged
# ---------------------------------------------------------------------------


def test_keepalive_chunks_flow_while_claude_writes_a_tool_call(fake_cli, monkeypatch):
    monkeypatch.setattr(ccc, "_KEEPALIVE_CHUNK_SECONDS", 0.05)
    monkeypatch.setattr(ccs, "_TICK_INTERVAL_SECONDS", 0.01)
    call = _call(name="write_file", arguments={"path": "/tmp/x", "content": "y" * 400})
    fake_cli.script({"text": "Writing.\n\n" + call, "delay": 0.01, "chunk": 8})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, calls, finish, empty = _stream(
        client, model="opus", messages=[{"role": "user", "content": "write"}],
        tools=[{"type": "function", "function": {"name": "write_file", "parameters": {}}}],
    )
    assert content == "Writing."
    assert [c.function.name for c in calls] == ["write_file"] and finish == ["tool_calls"]
    assert empty >= 2
    client.close()


def test_runtime_activity_reports_the_tool_being_written(fake_cli):
    call = _call(name="write_file", arguments={"path": "/tmp/x", "content": "y" * 600})
    fake_cli.script({
        "text": "Writing it.\n\n" + call, "delay": 0.01, "chunk": 6,
        "system": [{"subtype": "api_retry", "attempt": 2, "max_retries": 10,
                    "retry_delay_ms": 4000, "error_status": 529, "error": "overloaded"}],
    })
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    seen = []
    done = threading.Event()

    def poll():
        while not done.is_set():
            snapshot = client.get_runtime_activity()
            if snapshot and snapshot["active"]:
                seen.append(snapshot["description"])
            time.sleep(0.01)

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        _stream(client, model="opus", messages=[{"role": "user", "content": "write"}],
                tools=[{"type": "function", "function": {"name": "write_file", "parameters": {}}}])
    finally:
        done.set()
        poller.join()
    assert any("Claude is writing a write_file call" in text for text in seen)
    assert any("Anthropic API retry 2/10 (529), next in 4s" in text for text in seen)
    assert client.get_runtime_activity()["active"] is False
    client.close()


def test_progress_describes_thinking_and_waiting():
    progress = _Progress()
    assert progress.snapshot()["active"] is False
    progress.begin("starting")
    assert progress.snapshot()["description"].startswith("starting Claude Code")
    progress.update("thinking", tokens=120)
    assert progress.snapshot()["description"] == "Claude is thinking (0s, ~120 tokens)"
    progress.update("tool", chars=12400, calls=1, name="patch")
    assert progress.snapshot()["description"] == "Claude is writing a patch call (12.4k chars)"
    progress.notice("answered by claude-opus-4-8 after a model refusal fallback")
    assert progress.snapshot()["description"].endswith("after a model refusal fallback")
    progress.end()
    assert progress.snapshot() == {"active": False, "description": "", "updated_at": pytest.approx(time.time(), abs=5)}


def test_compaction_forgets_the_claude_session(fake_cli, caplog):
    fake_cli.script(
        {"text": "One."},
        {"text": "Two.", "system": [{"subtype": "compact_boundary",
                                      "compact_metadata": {"trigger": "auto", "pre_tokens": 967000}}]},
        {"text": "Three."},
    )
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "1"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "One."}, {"role": "user", "content": "2"}]
    with caplog.at_level(logging.WARNING, logger="agent.claude_code_session"):
        assert _run(session, fake_cli, history)[0] == "Two."
    assert "compacted its conversation" in caplog.text
    assert session._session_id is None
    history += [{"role": "assistant", "content": "Two."}, {"role": "user", "content": "3"}]
    assert _run(session, fake_cli, history)[0] == "Three."
    assert fake_cli.events("turn")[-1]["content"].startswith("FULL PROMPT")
    session.shutdown()


def test_session_wide_model_fallback_is_reported_and_not_kept_warm(fake_cli, caplog):
    fake_cli.script(
        {"text": "Answered.", "system": [{"subtype": "model_refusal_fallback", "trigger": "refusal",
                                          "originalModel": "claude-opus-5",
                                          "fallbackModel": "claude-opus-4-8"}]},
        {"text": "Again."},
    )
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "1"}]
    with caplog.at_level(logging.WARNING, logger="agent.claude_code_session"):
        _run(session, fake_cli, history)
    assert "model_refusal_fallback: claude-opus-5 -> claude-opus-4-8" in caplog.text
    assert session.last_usage["served_model"] == "claude-opus-4-8"
    history += [{"role": "assistant", "content": "Answered."}, {"role": "user", "content": "2"}]
    _run(session, fake_cli, history)
    # A new process resumes on the requested model.
    assert len(fake_cli.events("spawn")) == 2
    session.shutdown()


# ---------------------------------------------------------------------------
# Correct turn text from multi-message turns; stop_reason -> finish_reason
# ---------------------------------------------------------------------------


def _assistant(message_id, text, uuid_index, stop_reason=None):
    message = {"id": message_id, "content": [{"type": "text", "text": text}]}
    if stop_reason:
        message["stop_reason"] = stop_reason
    return json.dumps({"type": "assistant", "uuid": f"{uuid_index:08d}-1111-2222-3333-444444444444",
                       "session_id": SID, "message": message})


def _result(text, **extra):
    return json.dumps(dict({"type": "result", "subtype": "success", "is_error": False,
                            "session_id": SID, "result": text}, **extra))


def test_output_limit_continuation_joins_the_messages():
    head = 'Writing the report.\n\n<tool_call>{"id": "c1", "name": "write_file", "arguments": {"path": "/tmp/r.md", "con'
    tail = 'tent": "# Report"}}</tool_call>'
    stdout = "\n".join([_assistant("m1", head, 1, "max_tokens"), _assistant("m2", tail, 2), _result(tail)])
    response, _reasoning, _sid = _parse_stream_json_output(stdout)
    assert response == head + tail
    assert len(_parse_claude_reply(response).executable_calls) == 1
    # Stop reasons seen on the stream decide when the events carry none.
    stdout = "\n".join([_assistant("m1", head, 1), _assistant("m2", tail, 2), _result(tail)])
    assert _parse_stream_json_output(stdout)[0] == tail
    assert _parse_stream_json_output(stdout, stop_reasons={"m1": "max_tokens"})[0] == head + tail


def test_replaced_messages_keep_the_last_one():
    stdout = "\n".join([
        _assistant("m1", "An abandoned partial ans", 1),
        _assistant("m2", "The full answer.", 2, "end_turn"),
        _result("The full answer."),
    ])
    assert _parse_stream_json_output(stdout, stop_reasons={"m1": None})[0] == "The full answer."


def test_output_limit_split_tool_call_end_to_end(fake_cli):
    call = _call(name="write_file", arguments={"path": "/tmp/r.md", "content": "# Report " * 30})
    text = "Writing the report.\n\n" + call
    fake_cli.script({"text": text, "split": len(text) // 2})
    client = ClaudeCodeClient(command=fake_cli.command, cwd="/tmp")
    content, calls, finish, _ = _stream(
        client, model="opus", messages=[{"role": "user", "content": "report"}],
        tools=[{"type": "function", "function": {"name": "write_file", "parameters": {}}}],
    )
    assert content == "Writing the report."
    assert [c.function.name for c in calls] == ["write_file"] and finish == ["tool_calls"]
    assert "<tool_call" not in content
    client.close()


def test_stop_reasons_map_to_hermes_finish_reasons():
    assert _completion_parts("Partial answer", [], tools_offered=True, stop_reason="max_tokens")[2] == "length"
    assert _completion_parts("I can't help", [], tools_offered=False, stop_reason="refusal")[2] == "content_filter"
    assert _completion_parts("Fine.", [], tools_offered=True, stop_reason="end_turn")[2] == "stop"
    calls, _content, finish = _completion_parts(_call(), [], tools_offered=True, stop_reason="max_tokens")
    assert len(calls) == 1 and finish == "tool_calls"
    usage = _parse_stream_json_usage(_result("x", stop_reason="refusal", terminal_reason="completed",
                                             ttft_ms=812, usage={"output_tokens": 3}))
    assert usage["stop_reason"] == "refusal" and usage["terminal_reason"] == "completed"
    assert usage["ttft_ms"] == 812


# ---------------------------------------------------------------------------
# gate-quadratic-cost / stream-gate-quadratic
# ---------------------------------------------------------------------------


def test_gate_cost_is_linear_in_the_answer():
    out = []
    gate = _StreamGate(out.append, commit_chars=40)
    answer = ("Long prose paragraph with `inline code` and details. " * 4000)[:200_000]
    started = time.perf_counter()
    _feed(gate, answer, step=12)
    elapsed = time.perf_counter() - started
    gate.finish(answer)
    assert "".join(out) == answer.strip()
    # ~0.03 s now; the old gate rescanned the whole buffer per delta (~10 s).
    assert elapsed < 1.0, f"gate took {elapsed:.2f}s for 200k chars"


def test_gate_finds_tags_split_across_deltas():
    out = []
    gate = _StreamGate(out.append, commit_chars=10)
    text = "A committed answer that is long enough. " + _call()
    for index in range(len(text)):
        gate.feed(text[index])
    assert "".join(out) == "A committed answer that is long enough."
    assert "<" not in "".join(out)


# ---------------------------------------------------------------------------
# usage-lost-on-retries
# ---------------------------------------------------------------------------


def test_retried_past_attempts_are_accounted_separately(monkeypatch):
    session = ClaudeCodeSession()
    turns = iter([
        ("I'll check the logs now.", {"input_tokens": 40, "cached_tokens": 29000, "cache_write_tokens": 1000,
                                      "output_tokens": 25, "prompt_tokens": 30040, "completion_tokens": 25}),
        ("All logs are clean.", {"input_tokens": 50, "cached_tokens": 30000, "cache_write_tokens": 400,
                                 "output_tokens": 300, "prompt_tokens": 30450, "completion_tokens": 300}),
    ])

    def fake_execute(prompt, **kwargs):
        text, usage = next(turns)
        session._last_usage = dict(usage)
        session._last_turn_checkpoint = "aaaaaaaa-1111-2222-3333-444444444444"
        return text, "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    client = ClaudeCodeClient(cwd="/tmp")
    client._claude_session = session
    completion = client._create_chat_completion(
        model="opus", messages=[{"role": "user", "content": "logs?"}], tools=TOOLS
    )
    assert completion.choices[0].message.content == "All logs are clean."
    usage = completion.usage
    assert usage.prompt_tokens == 30450  # the context-size signal stays the final attempt
    assert usage.hermes_retry_usage.prompt_tokens == 30040
    assert usage.hermes_retry_usage.output_tokens == 25
    assert session.last_retry_usage["attempts"] == 1


# ---------------------------------------------------------------------------
# no-latency-telemetry
# ---------------------------------------------------------------------------


def test_each_turn_and_request_is_logged(fake_cli, caplog):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.INFO, logger="agent.claude_code_session"):
        _run(session, fake_cli, history, on_text_chunk=lambda _t: None)
        history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "more"}]
        _run(session, fake_cli, history, on_text_chunk=lambda _t: None)
    turns = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Claude Code turn:")]
    assert "mode=spawn-fresh" in turns[0] and "mode=warm" in turns[1]
    assert "ttft_ms=7" in turns[0] and "api_ms=12" in turns[0] and "outcome=ok" in turns[0]
    assert "first_event_ms=" in turns[0] and "text_ms=" in turns[0]
    requests = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Claude Code request:")]
    assert requests[0].startswith("Claude Code request: plan=fresh turns=1")
    assert requests[1].startswith("Claude Code request: plan=advance turns=1")
    session.shutdown()
