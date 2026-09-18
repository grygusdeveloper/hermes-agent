"""Claude Code bridge: native images, live streaming gate, warm processes."""

from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agent.claude_code_client import ClaudeCodeClient, _build_claude_code_request
from agent.claude_code_session import (
    _MAX_IMAGES_PER_REQUEST,
    ClaudeCodeSession,
    _StreamGate,
    _user_message_content,
    _WARM_IDLE,
)

PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
DATA_URL = f"data:image/png;base64,{PNG_B64}"
SID = "12345678-1234-1234-1234-123456789abc"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_request_moves_images_into_native_blocks():
    system_prompt, prompt, images = _build_claude_code_request(
        [
            {"role": "system", "content": "SYS"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            },
        ]
    )
    assert "SYS" in system_prompt and "SYS" not in prompt
    assert "[Image #1]" in prompt
    assert "image URL, not attached: https://example.com/a.png" in prompt
    assert images == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
    ]


def test_tool_result_images_are_forwarded():
    _sys, prompt, images = _build_claude_code_request(
        [
            {"role": "user", "content": "screenshot please"},
            {
                "role": "tool",
                "name": "browser_screenshot",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "captured"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            },
        ]
    )
    assert '<tool_result id="call_1" name="browser_screenshot">\ncaptured\n[Image #1]' in prompt
    assert len(images) == 1


def test_large_images_pass_through_for_the_cli_to_resize():
    """Claude Code resizes base64 image blocks itself; a 7 MB screenshot used
    to be replaced by a note and never reached Claude."""

    large = "A" * (7 * 1024 * 1024)
    _sys, prompt, images = _build_claude_code_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + large}},
                    {"type": "image_url", "image_url": {"url": "data:image/tiff;base64,AAAA"}},
                    {"type": "input_audio", "input_audio": {"data": "UklG"}},
                ],
            }
        ]
    )
    assert len(images) == 1 and images[0]["source"]["data"] == large
    assert "[Image #1]" in prompt
    assert "unsupported type image/tiff" in prompt
    assert "[non-text part omitted: type=input_audio]" in prompt


def test_absurd_image_payload_is_still_a_note():
    from agent.claude_code_session import _MAX_IMAGE_BASE64_CHARS

    huge = "data:image/png;base64," + "A" * (_MAX_IMAGE_BASE64_CHARS + 4)
    _sys, prompt, images = _build_claude_code_request(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": huge}}]}]
    )
    assert images == []
    assert "larger than 32 MB" in prompt


def test_user_content_keeps_only_most_recent_images():
    images = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": str(i)}}
        for i in range(_MAX_IMAGES_PER_REQUEST + 2)
    ]
    content = _user_message_content("transcript", images)
    assert content[0] == {"type": "text", "text": "transcript"}
    sent = [block for block in content if block["type"] == "image"]
    assert len(sent) == _MAX_IMAGES_PER_REQUEST
    assert sent[0]["source"]["data"] == "2"
    assert "Images #1-#2 are not re-sent" in content[1]["text"]
    assert _user_message_content("plain", []) == "plain"


def test_session_run_sends_image_blocks_on_stdin(monkeypatch):
    session = ClaudeCodeSession()
    seen = {}

    def fake_execute(prompt, **kwargs):
        seen["prompt"] = prompt
        return "It is a logo.", "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    client = ClaudeCodeClient(cwd="/tmp")
    client._claude_session = session
    client._create_chat_completion(
        model="sonnet",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            }
        ],
    )
    assert isinstance(seen["prompt"], list)
    assert seen["prompt"][0]["type"] == "text"
    assert seen["prompt"][-1]["type"] == "image"


def test_claude_code_models_report_native_vision():
    from agent.image_routing import _lookup_supports_vision

    assert _lookup_supports_vision("claude-code", "claude-opus-5", {}) is True
    assert _lookup_supports_vision("claude-code", "opus", {}) is True


# ---------------------------------------------------------------------------
# Streaming gate
# ---------------------------------------------------------------------------


def test_gate_holds_short_answers_until_validated():
    out = []
    gate = _StreamGate(out.append)
    gate.feed("Running fine — nothing pending.")
    assert out == []
    gate.finish("Running fine — nothing pending.")
    assert out == ["Running fine — nothing pending."]


def test_gate_streams_long_prose_and_never_tool_markup():
    out = []
    gate = _StreamGate(out.append, commit_chars=40)
    prose = "Here is a long explanation of the change. " * 3
    call = '<tool_call>{"id":"c1","type":"function","function":{"name":"t","arguments":"{}"}}</tool_call>'
    raw = prose + call
    for i in range(0, len(raw), 7):
        gate.feed(raw[i : i + 7])
    assert gate.committed
    assert out, "long prose should stream before the answer completes"
    gate.finish(raw)
    joined = "".join(out)
    assert "<tool_call" not in joined
    assert joined == prose.strip()


def test_retry_wrapper_streams_only_the_validated_attempt(monkeypatch):
    session = ClaudeCodeSession()
    long_answer = "Findings: " + "the provider is healthy and well configured. " * 20
    attempts = iter(["Checking the logs.", long_answer])

    def fake_execute(prompt, *, on_event=None, **kwargs):
        text = next(attempts)
        if on_event is not None:
            for i in range(0, len(text), 11):
                on_event("text", text[i : i + 11])
            on_event("thinking", "considering")
        return text, "considering", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    chunks, thoughts = [], []
    response, _r, _sid = session._execute_with_soft_limit_retry(
        "prompt",
        session_id=None,
        model="claude-opus-5",
        effort="high",
        timeout_seconds=30,
        cwd="/tmp",
        env={},
        on_text_chunk=chunks.append,
        on_reasoning_chunk=thoughts.append,
        had_tools=True,
    )
    assert response == long_answer
    assert "Checking the logs." not in "".join(chunks)
    assert "".join(chunks) == long_answer.strip()
    assert len(chunks) > 1
    assert thoughts == ["considering", "considering"]


def test_client_stream_reasoning_is_not_duplicated(monkeypatch):
    client = ClaudeCodeClient(cwd="/tmp")

    def fake_run(prompt, *, on_text_chunk=None, on_reasoning_chunk=None, **kwargs):
        on_reasoning_chunk("thinking hard")
        on_text_chunk("Answer.")
        return "Answer.", "thinking hard"

    monkeypatch.setattr(client._claude_session, "run", fake_run)
    chunks = list(
        client._create_chat_completion(
            model="sonnet", messages=[{"role": "user", "content": "q"}], stream=True
        )
    )
    reasoning = [
        c.choices[0].delta.reasoning_content
        for c in chunks
        if c.choices and c.choices[0].delta.reasoning_content
    ]
    content = "".join(
        c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
    )
    assert reasoning == ["thinking hard"]
    assert content == "Answer."


# ---------------------------------------------------------------------------
# Warm processes (fake stream-json CLI)
# ---------------------------------------------------------------------------

FAKE_CLI = textwrap.dedent(
    """
    import json, os, sys, uuid
    log = os.environ["FAKE_CLAUDE_LOG"]
    argv = sys.argv[1:]
    sid = None
    if "--resume" in argv:
        sid = argv[argv.index("--resume") + 1]
    elif "--session-id" in argv:
        sid = argv[argv.index("--session-id") + 1]
    with open(log, "a") as fh:
        fh.write(json.dumps({"event": "spawn", "argv": argv}) + "\\n")
    cost = 0.0
    for line in sys.stdin:
        msg = json.loads(line)
        content = msg["message"]["content"]
        with open(log, "a") as fh:
            fh.write(json.dumps({"event": "turn", "content": content}) + "\\n")
        text = os.environ.get("FAKE_CLAUDE_REPLY", "Warm reply.")
        cost += 0.01
        def out(obj):
            sys.stdout.write(json.dumps(obj) + "\\n")
        out({"type": "system", "subtype": "init", "session_id": sid})
        out({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "resetsAt": 1}})
        if "--include-partial-messages" in argv:
            for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
                out({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": piece}}})
        out({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
        out({"type": "result", "subtype": "success", "is_error": False, "result": text,
             "session_id": sid, "total_cost_usd": cost,
             "usage": {"input_tokens": 3, "output_tokens": 5}})
        sys.stdout.flush()
    """
)


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI)
    launcher = tmp_path / "claude"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")

    def events():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    yield str(launcher), events
    for warm in list(_WARM_IDLE.values()):
        warm.close()


def _run(session, command, messages, **kwargs):
    return session.run(
        "FULL PROMPT",
        messages=messages,
        model="sonnet",
        tools_digest="digest",
        timeout_seconds=30,
        cwd="/tmp",
        env={"PATH": os.environ.get("PATH", ""), "FAKE_CLAUDE_LOG": os.environ["FAKE_CLAUDE_LOG"]},
        state_key="conv-1",
        command=command,
        system_prompt="SYSTEM CONTRACT",
        **kwargs,
    )


def test_warm_process_serves_the_tool_loop_without_respawning(fake_cli):
    command, events = fake_cli
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    assert _run(session, command, history)[0] == "Warm reply."
    history += [
        {"role": "assistant", "content": "Warm reply."},
        {"role": "tool", "name": "terminal", "tool_call_id": "c1", "content": "ok"},
    ]
    chunks = []
    assert _run(session, command, history, on_text_chunk=chunks.append)[0] == "Warm reply."
    spawns = [e for e in events() if e["event"] == "spawn"]
    turns = [e for e in events() if e["event"] == "turn"]
    assert len(spawns) == 1
    assert "--system-prompt-file" in spawns[0]["argv"]
    assert turns[0]["content"] == "FULL PROMPT"
    assert '<tool_result id="c1" name="terminal">\nok\n</tool_result>' in turns[1]["content"]
    assert "".join(chunks) == "Warm reply."
    # total_cost_usd is cumulative per process; each turn reports its share.
    assert session.last_usage["total_cost_usd"] == pytest.approx(0.01)
    assert session.last_rate_limit["status"] == "allowed"
    session.shutdown()


def test_keepalive_disabled_resumes_in_a_new_process(fake_cli, monkeypatch):
    command, events = fake_cli
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, command, history)
    history += [{"role": "assistant", "content": "Warm reply."}, {"role": "user", "content": "more"}]
    _run(session, command, history)
    spawns = [e for e in events() if e["event"] == "spawn"]
    assert len(spawns) == 2
    assert "--resume" in spawns[1]["argv"]


def test_warm_process_that_died_while_idle_is_resumed(fake_cli):
    command, events = fake_cli
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, command, history)
    warm = session._warm
    assert warm is not None
    warm.process.kill()
    warm.process.wait(timeout=5)
    history += [{"role": "assistant", "content": "Warm reply."}, {"role": "user", "content": "again"}]
    assert _run(session, command, history)[0] == "Warm reply."
    spawns = [e for e in events() if e["event"] == "spawn"]
    assert len(spawns) == 2
    assert "--resume" in spawns[1]["argv"]
    session.shutdown()


def test_warm_process_is_dropped_when_history_diverges(fake_cli):
    command, events = fake_cli
    session = ClaudeCodeSession()
    _run(session, command, [{"role": "user", "content": "hi"}])
    # /new or compression: a different history cannot reuse the warm process.
    _run(session, command, [{"role": "user", "content": "brand new topic"}])
    spawns = [e for e in events() if e["event"] == "spawn"]
    assert len(spawns) == 2
    assert "--session-id" in spawns[1]["argv"]
    session.shutdown()


def test_idle_warm_process_expires(fake_cli, monkeypatch):
    command, _events = fake_cli
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0.3")
    session = ClaudeCodeSession()
    _run(session, command, [{"role": "user", "content": "hi"}])
    warm = session._warm
    assert warm is not None
    deadline = time.time() + 10
    while warm.process.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    assert warm.process.poll() is not None


def test_gate_commits_early_without_tools_even_for_limit_wording():
    out = []
    gate = _StreamGate(out.append, had_tools=False)
    answer = "Sure — here is a short poem about the sea and the patient light of morning. " * 2
    for i in range(0, len(answer), 5):
        gate.feed(answer[i : i + 5])
    assert gate.committed and out

    # Live text deltas are model output; CLI banners never arrive this way, so
    # an answer *about* rate limits streams like any other.
    limit_out = []
    limits = _StreamGate(limit_out.append, had_tools=False)
    about = "HTTP 429 means Too Many Requests: the server is rate limiting you. " * 2
    limits.feed(about)
    assert limits.committed and limit_out


def test_committed_stream_is_accepted_even_if_it_mentions_limits(monkeypatch):
    session = ClaudeCodeSession()
    answer = (
        "Your gateway logs show the worker restarted twice overnight and then settled. "
        "Later a client hit a usage limit reached error on the paid tier."
    )
    calls = {"n": 0}

    def fake_execute(prompt, *, on_event=None, **kwargs):
        calls["n"] += 1
        for i in range(0, len(answer), 9):
            on_event("text", answer[i : i + 9])
        return answer, "", SID

    monkeypatch.setattr(session, "_execute", fake_execute)
    chunks = []
    response, _r, _s = session._execute_with_soft_limit_retry(
        "p", session_id=None, model="m", effort=None, timeout_seconds=5,
        cwd="/tmp", env={}, on_text_chunk=chunks.append, had_tools=False,
    )
    assert calls["n"] == 1
    assert "".join(chunks) == answer
