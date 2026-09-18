"""Claude Code bridge continuity: fingerprints, per-agent keys, checkpoints.

Most tests drive the real ``ClaudeCodeSession`` against a fake stream-json CLI
that behaves like Claude Code 2.1.x where it matters here: it persists every
user entry it reads (unless ``--no-session-persistence``), gives each
assistant entry a chain ``uuid``, honours ``--resume`` and
``--resume-session-at`` (truncating the loaded chain), and logs the context
the "model" sees on each turn.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from agent.claude_code_client import ClaudeCodeClient, _build_claude_code_request
from agent.claude_code_session import (
    _ASSISTANT_ONLY_CONTINUATION_PROMPT,
    _PROGRESS_CONTINUATION_PROMPT,
    _RESEND_REPAIR_PROMPT,
    _STATE_VERSION,
    ClaudeCodeSession,
    _incremental_prompt_with_images,
    _load_durable_state,
    _message_fingerprint,
    _prune_durable_states,
    _last_assistant_uuid,
    _save_durable_state,
    _state_dir,
    _state_path,
    _WARM_IDLE,
)

SID = "12345678-1234-1234-1234-123456789abc"
UUID_A = "aaaaaaaa-1111-2222-3333-444444444444"

FAKE_CLI = textwrap.dedent(
    """
    import json, os, sys, time, uuid
    log = os.environ["FAKE_CLAUDE_LOG"]
    store = os.environ["FAKE_CLAUDE_STORE"]
    script = os.environ.get("FAKE_CLAUDE_SCRIPT", "")
    argv = sys.argv[1:]

    def arg(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    def write_log(obj):
        with open(log, "a") as fh:
            fh.write(json.dumps(obj) + "\\n")

    def next_reply():
        if not script or not os.path.exists(script):
            return {}
        counter = script + ".n"
        n = int(open(counter).read()) if os.path.exists(counter) else 0
        replies = json.load(open(script))
        with open(counter, "w") as fh:
            fh.write(str(n + 1))
        return replies[n] if n < len(replies) else {}

    sid = arg("--resume") or arg("--session-id")
    resume_at = arg("--resume-session-at")
    persist = "--no-session-persistence" not in argv
    path = os.path.join(store, sid + ".json")
    chain = []
    write_log({"event": "spawn", "argv": argv})
    if resume_at and os.environ.get("FAKE_CLAUDE_REJECT_RESUME_AT"):
        sys.stderr.write("error: unknown option '--resume-session-at'\\n")
        sys.exit(1)
    if "--resume" in argv:
        if not os.path.exists(path):
            sys.stderr.write("No conversation found with session ID: " + sid + "\\n")
            sys.exit(1)
        chain = json.load(open(path))
        if resume_at:
            idx = next((i for i, e in enumerate(chain) if e["uuid"] == resume_at), None)
            if idx is None:
                sys.stderr.write("No message found with message.uuid of: " + resume_at + "\\n")
                sys.exit(1)
            chain = chain[: idx + 1]

    def save():
        if persist:
            with open(path, "w") as fh:
                json.dump(chain, fh)

    def out(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")

    for line in sys.stdin:
        content = json.loads(line)["message"]["content"]
        text_in = content if isinstance(content, str) else json.dumps(content)
        # Claude Code persists the user entry before calling the API.
        chain.append({"uuid": str(uuid.uuid4()), "text": text_in})
        save()
        write_log({"event": "turn", "content": content, "context": [e["text"] for e in chain]})
        reply = next_reply()
        time.sleep(float(reply.get("sleep", os.environ.get("FAKE_CLAUDE_SLEEP", "0"))))
        out({"type": "system", "subtype": "init", "session_id": sid})
        if reply.get("error"):
            out({"type": "result", "subtype": "error_during_execution", "is_error": True,
                 "result": "API Error: 500 boom", "session_id": sid})
            sys.stdout.flush()
            continue
        if reply.get("rate_limit"):
            out({"type": "result", "subtype": "error_during_execution", "is_error": True,
                 "result": "API Error: 429 rate_limit", "api_error_status": 429,
                 "session_id": sid})
            sys.stdout.flush()
            continue
        text = reply.get("text", os.environ.get("FAKE_CLAUDE_REPLY", "Fake reply."))
        thinking_uuid, text_uuid = str(uuid.uuid4()), str(uuid.uuid4())
        # Older CLIs emit no chain uuid on assistant events.
        shown = {} if reply.get("no_uuid") else {"uuid": thinking_uuid}
        out(dict(shown, type="assistant", session_id=sid,
                 message={"content": [{"type": "thinking", "thinking": ""}]}))
        shown = {} if reply.get("no_uuid") else {"uuid": text_uuid}
        out(dict(shown, type="assistant", session_id=sid,
                 message={"content": [{"type": "text", "text": text}]}))
        chain.append({"uuid": thinking_uuid, "text": "(thinking)"})
        chain.append({"uuid": text_uuid, "text": "A: " + text})
        save()
        out({"type": "result", "subtype": "success", "is_error": False, "result": text,
             "session_id": sid, "usage": {"input_tokens": 1, "output_tokens": 1}})
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
    store = tmp_path / "store"
    store.mkdir()
    log = tmp_path / "calls.jsonl"
    replies = tmp_path / "replies.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_STORE", str(store))
    monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", str(replies))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "30")

    def events(kind=None):
        if not log.exists():
            return []
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        return [row for row in rows if kind is None or row["event"] == kind]

    def script_replies(*items):
        replies.write_text(json.dumps(list(items)))
        counter = replies.with_name(replies.name + ".n")
        if counter.exists():
            counter.unlink()

    yield SimpleNamespace(command=str(launcher), events=events, script=script_replies)
    for warm in list(_WARM_IDLE.values()):
        warm.close()


def _env():
    keys = ("PATH", "FAKE_CLAUDE_LOG", "FAKE_CLAUDE_STORE", "FAKE_CLAUDE_SCRIPT",
            "FAKE_CLAUDE_REPLY", "FAKE_CLAUDE_SLEEP", "FAKE_CLAUDE_REJECT_RESUME_AT")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _run(session, cli, messages, *, state_key="root|main", system_prompt="SYSTEM", **kwargs):
    return session.run(
        "FULL PROMPT",
        messages=messages,
        model="sonnet",
        tools_digest="digest",
        timeout_seconds=30,
        cwd="/tmp",
        env=_env(),
        state_key=state_key,
        command=cli.command,
        system_prompt=system_prompt,
        **kwargs,
    )


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _tool_call(call_id, name="terminal", arguments='{"command":"ls"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ---------------------------------------------------------------------------
# Fingerprints survive the session-DB reload; renderings keep tool names
# ---------------------------------------------------------------------------


def _db_reloaded(messages):
    """What the gateway sends on the next turn: DB rows -> chat-completions."""

    from agent.transports.chat_completions import ChatCompletionsTransport

    rows = []
    for message in messages:
        row = {k: v for k, v in message.items() if k != "name"}
        if message.get("role") == "tool" and message.get("name"):
            row["tool_name"] = message["name"]
        row["timestamp"] = 1_700_000_000.0
        rows.append(row)
    return ChatCompletionsTransport().convert_messages(rows)


def test_fingerprint_matches_db_reloaded_tool_message():
    from agent.tool_dispatch_helpers import make_tool_result_message
    from agent.transports.chat_completions import ChatCompletionsTransport

    live = ChatCompletionsTransport().convert_messages(
        [make_tool_result_message("terminal", "ok\n", "k1")]
    )
    assert live[0].get("name") == "terminal"
    reloaded = _db_reloaded(live)
    assert "name" not in reloaded[0] and "tool_name" not in reloaded[0]
    assert _message_fingerprint(live) == _message_fingerprint(reloaded)
    # Error stubs are saved with tool_name NULL: they come back with neither.
    assert _message_fingerprint(
        [{"role": "tool", "name": "bogus", "tool_call_id": "x", "content": "E"}]
    ) == _message_fingerprint([{"role": "tool", "tool_call_id": "x", "content": "E"}])
    # Content and linkage still count.
    assert _message_fingerprint(
        [{"role": "tool", "tool_call_id": "x", "content": "E"}]
    ) != _message_fingerprint([{"role": "tool", "tool_call_id": "y", "content": "E"}])


def test_renderings_restore_tool_names_from_assistant_calls():
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    _system, prompt, _images = _build_claude_code_request(messages)
    assert "Tool Result (name=terminal, tool_call_id=c1):\na.txt" in prompt
    incremental, _ = _incremental_prompt_with_images(messages, 2)
    assert incremental == "Tool Result (name=terminal, tool_call_id=c1):\na.txt"


def test_reused_call_ids_name_results_after_the_preceding_call():
    """Regression: Claude reuses short ids (``g1``) across turns; a last-wins
    id map labelled an old read_file result as the newest call's terminal."""

    messages = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("g1", "read_file")]},
        {"role": "tool", "tool_call_id": "g1", "content": "file body"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "run it twice"},
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("g1", "terminal"), _tool_call("g1", "process")]},
        {"role": "tool", "tool_call_id": "g1", "content": "exit 0"},
        {"role": "tool", "tool_call_id": "g1", "content": "pid 7"},
    ]
    _system, prompt, _images = _build_claude_code_request(messages)
    headers = [line for line in prompt.splitlines() if line.startswith("Tool Result")]
    assert headers == [
        "Tool Result (name=read_file, tool_call_id=g1):",
        "Tool Result (name=terminal, tool_call_id=g1):",
        "Tool Result (name=process, tool_call_id=g1):",
    ]
    incremental, _ = _incremental_prompt_with_images(messages[:7], 3)
    assert "Tool Result (name=terminal, tool_call_id=g1):\nexit 0" in incremental
    incremental, _ = _incremental_prompt_with_images(messages, 7)
    assert incremental == "Tool Result (name=process, tool_call_id=g1):\npid 7"


def test_state_version_bump_ignores_old_states(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fingerprints = _message_fingerprint([{"role": "user", "content": "hi"}])
    _save_durable_state(
        "k", SID, fingerprints, model="sonnet", effort=None, tools_digest="d",
        system_digest="0" * 64, checkpoints=((1, UUID_A),),
    )
    state = _load_durable_state("k")
    assert state.checkpoints == ((1, UUID_A),) and state.system_digest == "0" * 64
    path = _state_path("k")
    payload = json.loads(path.read_text())
    payload["version"] = _STATE_VERSION - 1
    path.write_text(json.dumps(payload))
    assert _load_durable_state("k") is None


def test_gateway_reload_resumes_with_only_the_new_user_message(fake_cli, monkeypatch):
    """Regression: every user turn after tool use used to replay everything."""

    from agent.tool_dispatch_helpers import make_tool_result_message
    from agent.transports.chat_completions import ChatCompletionsTransport

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")  # force a cold resume
    live_turn = ChatCompletionsTransport().convert_messages(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1")]},
            make_tool_result_message("terminal", "a.txt", "c1"),
        ]
    )
    session = ClaudeCodeSession()
    _run(session, fake_cli, live_turn)
    next_turn = _db_reloaded(live_turn) + [
        {"role": "assistant", "content": "Fake reply."},
        {"role": "user", "content": "thanks, now count them"},
    ]
    _run(ClaudeCodeSession(), fake_cli, next_turn)
    spawns, turns = fake_cli.events("spawn"), fake_cli.events("turn")
    assert len(spawns) == 2
    assert _arg(spawns[1]["argv"], "--resume") == _arg(spawns[0]["argv"], "--session-id")
    assert turns[1]["content"] == "User:\nthanks, now count them"


# ---------------------------------------------------------------------------
# Per-agent bridge state keys
# ---------------------------------------------------------------------------


def test_bridge_state_key_scoping():
    from agent.portal_tags import (
        get_bridge_state_key,
        reset_bridge_state_key,
        reset_conversation_context,
        set_bridge_state_key,
        set_conversation_context,
    )

    conversation = set_conversation_context("root")
    try:
        # Nothing published: the legacy conversation id.
        assert get_bridge_state_key() == "root"
        token = set_bridge_state_key(None)
        assert get_bridge_state_key() is None
        reset_bridge_state_key(token)
        holder = {"sid": "s1"}
        token = set_bridge_state_key(lambda: f"root|{holder['sid']}")
        assert get_bridge_state_key() == "root|s1"
        holder["sid"] = "s2"  # compression rotation mid-turn
        assert get_bridge_state_key() == "root|s2"
        reset_bridge_state_key(token)
        assert get_bridge_state_key() == "root"
    finally:
        reset_conversation_context(conversation)


def test_bridge_state_key_resolver_per_agent():
    from run_agent import _bridge_state_key_resolver

    main = SimpleNamespace(session_id="main-sid")
    child = SimpleNamespace(session_id="child-sid", platform="subagent")
    fork = SimpleNamespace(session_id="main-sid", _persist_disabled=True)
    assert _bridge_state_key_resolver(main, "root")() == "root|main-sid"
    assert _bridge_state_key_resolver(child, "root")() == "root|child-sid"
    assert _bridge_state_key_resolver(fork, "root") is None
    assert _bridge_state_key_resolver(SimpleNamespace(session_id=None), "root")() is None
    assert _bridge_state_key_resolver(SimpleNamespace(session_id="solo"), None)() == "solo|solo"


@pytest.mark.parametrize("persist_disabled", [False, True])
def test_run_conversation_publishes_and_restores_bridge_key(monkeypatch, persist_disabled):
    from agent.portal_tags import get_bridge_state_key, get_conversation_context
    from run_agent import AIAgent

    seen = {}

    def fake_loop(agent, *_args, **_kwargs):
        seen["conversation"] = get_conversation_context()
        seen["bridge"] = get_bridge_state_key()
        return {"final_response": "ok", "completed": True}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_loop)
    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        model="m",
        _parent_session_id=None,
        _session_db=None,
        _persist_disabled=persist_disabled,
        _conversation_root_id=lambda: "root",
    )
    AIAgent.run_conversation(agent, "hi", task_id="t")
    assert seen["conversation"] == "root"
    assert seen["bridge"] == (None if persist_disabled else "root|s1")
    assert get_bridge_state_key() is None  # restored: nothing ambient


def test_client_passes_bridge_key_and_tool_loop_keepalive():
    from agent.portal_tags import reset_bridge_state_key, set_bridge_state_key

    client = ClaudeCodeClient(cwd="/tmp")
    calls = []

    def fake_run(prompt_text, **kwargs):
        calls.append(kwargs)
        return "ok", ""

    client._claude_session.run = fake_run
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    messages = [{"role": "user", "content": "hi"}]
    token = set_bridge_state_key("root|main")
    try:
        client._create_chat_completion(model="sonnet", messages=messages, tools=tools)
        client._create_chat_completion(model="sonnet", messages=messages)
    finally:
        reset_bridge_state_key(token)
    token = set_bridge_state_key(None)  # review fork
    try:
        client._create_chat_completion(model="sonnet", messages=messages, tools=tools)
    finally:
        reset_bridge_state_key(token)
    assert [(c["state_key"], c["keepalive"]) for c in calls] == [
        ("root|main", True),
        (None, False),
        (None, True),
    ]


def test_sibling_agent_keys_do_not_reset_each_other(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    main, child = ClaudeCodeSession(), ClaudeCodeSession()
    main_history = [{"role": "user", "content": "main task"}]
    child_history = [{"role": "user", "content": "child task"}]
    _run(main, fake_cli, main_history, state_key="root|main")
    _run(child, fake_cli, child_history, state_key="root|child")
    main_history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "main 2"}]
    child_history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "child 2"}]
    _run(main, fake_cli, main_history, state_key="root|main")
    _run(child, fake_cli, child_history, state_key="root|child")
    spawns, turns = fake_cli.events("spawn"), fake_cli.events("turn")
    assert ["--resume" in s["argv"] for s in spawns] == [False, False, True, True]
    assert [t["content"] for t in turns[2:]] == ["User:\nmain 2", "User:\nchild 2"]


def test_stateless_fork_leaves_parent_state_untouched(fake_cli):
    main = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(main, fake_cli, history, state_key="root|main")
    main_state = _state_path("root|main").read_bytes()

    fork = ClaudeCodeSession()
    review = history + [{"role": "assistant", "content": "Fake reply."},
                        {"role": "user", "content": "Review the conversation above"}]
    _run(fork, fake_cli, review, state_key=None, keepalive=True)
    review += [{"role": "assistant", "content": "", "tool_calls": [_tool_call("m1", "memory")]},
               {"role": "tool", "tool_call_id": "m1", "content": "saved"}]
    _run(fork, fake_cli, review, state_key=None, keepalive=True)

    assert _state_path("root|main").read_bytes() == main_state
    assert len(list(_state_dir().glob("*.json"))) == 1
    fork_spawns = fake_cli.events("spawn")[1:]
    assert len(fork_spawns) == 1, "the fork's tool loop reuses its warm process"
    assert "--no-session-persistence" in fork_spawns[0]["argv"]
    assert "--no-session-persistence" not in fake_cli.events("spawn")[0]["argv"]
    assert fake_cli.events("turn")[-1]["content"] == (
        "Tool Result (name=memory, tool_call_id=m1):\nsaved"
    )
    main.shutdown()
    fork.shutdown()


def test_concurrent_children_do_not_serialize(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "1.0")
    errors = []

    def child(index):
        try:
            session = ClaudeCodeSession()
            _run(session, fake_cli, [{"role": "user", "content": f"task {index}"}],
                 state_key=f"root|child-{index}", keepalive=False)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=child, args=(i,)) for i in range(3)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started
    assert not errors
    assert elapsed < 2.4, f"children serialized: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Stateless calls
# ---------------------------------------------------------------------------


def test_stateless_one_shot_is_not_persisted_or_cold_resumed(fake_cli):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "title this"}]
    _run(session, fake_cli, history, state_key=None)
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "again"}]
    _run(session, fake_cli, history, state_key=None)
    spawns, turns = fake_cli.events("spawn"), fake_cli.events("turn")
    assert all("--no-session-persistence" in s["argv"] for s in spawns)
    assert "--resume" not in spawns[1]["argv"]
    assert turns[1]["content"] == "FULL PROMPT"
    assert not _state_dir().exists() or not list(_state_dir().glob("*.json"))


def test_stateless_preamble_rerolls_instead_of_resuming(fake_cli):
    fake_cli.script({"text": "Checking the logs."}, {"text": "All logs are clean and rotated."})
    session = ClaudeCodeSession()
    response, _ = _run(session, fake_cli, [{"role": "user", "content": "check"}], state_key=None)
    assert response == "All logs are clean and rotated."
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2 and all("--resume" not in s["argv"] for s in spawns)
    assert [t["content"] for t in fake_cli.events("turn")] == ["FULL PROMPT", "FULL PROMPT"]


# ---------------------------------------------------------------------------
# Checkpoints: failed attempts never pollute the resumed chain
# ---------------------------------------------------------------------------


def _tool_turn(history, call_id="c1", output="a.txt"):
    return history + [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call(call_id)]},
        {"role": "tool", "name": "terminal", "tool_call_id": call_id, "content": output},
    ]


def test_checkpoint_is_the_last_assistant_entry():
    def event(uuid_value):
        return json.dumps({"type": "assistant", "uuid": uuid_value, "message": {"content": []}})

    thinking, text = UUID_A, SID
    stdout = "\n".join([event(thinking), event(text), '{"type":"result"}'])
    assert _last_assistant_uuid(stdout) == text
    # An unusable last uuid must not fall back to the thinking block: resuming
    # there would cut the reply off.
    assert _last_assistant_uuid("\n".join([event(thinking), event("not-a-uuid")])) is None
    assert _last_assistant_uuid('{"type":"result"}') is None


def test_cli_without_resume_session_at_degrades_to_plain_resume(monkeypatch):
    import agent.claude_code_session as ccs
    from agent.claude_code_session import ClaudeCodeSessionExpired

    monkeypatch.setattr(ccs, "_resume_at_supported", True)
    session = ClaudeCodeSession()

    def argv():
        return session._build_argv(
            "claude", session_id=SID, model="sonnet", effort=None,
            system_prompt=None, stream_partials=False, resume_at=UUID_A,
        )

    assert _arg(argv(), "--resume-session-at") == UUID_A
    with pytest.raises(ClaudeCodeSessionExpired):
        session._raise_process_failure(
            1, "", "error: unknown option '--resume-session-at'", SID
        )
    assert "--resume-session-at" not in argv() and _arg(argv(), "--resume") == SID


def test_cold_resume_passes_the_published_checkpoint(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    state = _load_durable_state("root|main")
    assert len(state.checkpoints) == 1 and state.checkpoints[0][0] == 1
    _run(session, fake_cli, _tool_turn(history))
    spawn = fake_cli.events("spawn")[1]
    assert _arg(spawn["argv"], "--resume-session-at") == state.checkpoints[0][1]
    assert [cp[0] for cp in _load_durable_state("root|main").checkpoints] == [1, 3]


def test_failed_attempt_is_not_duplicated_on_retry(fake_cli):
    fake_cli.script({}, {"error": True}, {})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history = _tool_turn(history)
    with pytest.raises(RuntimeError):
        _run(session, fake_cli, history)  # the warm process got the payload, then failed
    _run(session, fake_cli, history)  # Hermes retries the same request
    context = fake_cli.events("turn")[-1]["context"]
    assert sum("Tool Result" in entry for entry in context) == 1
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2 and "--resume-session-at" in spawns[1]["argv"]


def test_preamble_exhaustion_is_dropped_on_the_next_attempt(fake_cli):
    fake_cli.script({}, {"text": "Checking the logs."}, {"text": "Checking the logs."},
                    {"text": "Checking the logs."}, {"text": "Everything is fine."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history = _tool_turn(history)
    with pytest.raises(RuntimeError, match="intermediate planning"):
        _run(session, fake_cli, history)
    turns = fake_cli.events("turn")
    assert [t["content"] for t in turns[2:4]] == [_PROGRESS_CONTINUATION_PROMPT] * 2
    response, _ = _run(session, fake_cli, history)
    assert response == "Everything is fine."
    context = fake_cli.events("turn")[-1]["context"]
    assert sum("Tool Result" in entry for entry in context) == 1
    assert not any(_PROGRESS_CONTINUATION_PROMPT in entry for entry in context)
    assert len(fake_cli.events("spawn")) == 2


def test_soft_limit_retry_resumes_at_the_checkpoint(fake_cli, monkeypatch):
    monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
    banner = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"
    fake_cli.script({}, {"text": banner}, {"text": "The listing shows one file."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    response, _ = _run(session, fake_cli, _tool_turn(history))
    assert response == "The listing shows one file."
    context = fake_cli.events("turn")[-1]["context"]
    assert sum("Tool Result" in entry for entry in context) == 1
    assert not any(banner in entry for entry in context)


# ---------------------------------------------------------------------------
# Re-send, assistant-only delta, rewind
# ---------------------------------------------------------------------------


def test_identical_resend_is_repaired_in_session(fake_cli):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    before = _load_durable_state("root|main")
    _run(session, fake_cli, history)
    turns = fake_cli.events("turn")
    assert len(fake_cli.events("spawn")) == 1
    assert turns[1]["content"] == _RESEND_REPAIR_PROMPT
    after = _load_durable_state("root|main")
    assert after.fingerprints == before.fingerprints
    assert len(after.checkpoints) == 1 and after.checkpoints != before.checkpoints


def test_assistant_only_delta_continues_in_session(fake_cli):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    _run(session, fake_cli, history + [{"role": "assistant", "content": "Partial"}])
    assert len(fake_cli.events("spawn")) == 1
    assert fake_cli.events("turn")[1]["content"] == _ASSISTANT_ONLY_CONTINUATION_PROMPT


def test_rewound_history_resumes_at_the_older_checkpoint(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    session = ClaudeCodeSession()
    turn_a = [{"role": "system", "content": "rules"}, {"role": "user", "content": "u1"}]
    _run(session, fake_cli, turn_a)
    turn_b = turn_a + [{"role": "assistant", "content": "F1"}, {"role": "user", "content": "u2"}]
    _run(session, fake_cli, turn_b)
    _run(session, fake_cli, _tool_turn(turn_b))
    first_checkpoint = _load_durable_state("root|main").checkpoints[0]
    assert first_checkpoint[0] == 2

    # /retry of turn B: the tool loop is gone, u2 is re-sent.
    _run(session, fake_cli, turn_b)
    spawn, turn = fake_cli.events("spawn")[-1], fake_cli.events("turn")[-1]
    assert _arg(spawn["argv"], "--resume-session-at") == first_checkpoint[1]
    assert turn["content"] == "User:\nu2"
    assert not any("Tool Result" in entry for entry in turn["context"])
    assert [cp[0] for cp in _load_durable_state("root|main").checkpoints] == [2, 4]


def test_unknown_checkpoint_falls_back_to_a_fresh_session(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    state = _load_durable_state("root|main")
    _save_durable_state(
        "root|main", state.session_id, state.fingerprints, model=state.model,
        effort=state.effort, tools_digest=state.tools_digest,
        system_digest=state.system_digest, checkpoints=((1, UUID_A),),
    )
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "more"}]
    response, _ = _run(session, fake_cli, history)
    assert response == "Fake reply."
    spawns = fake_cli.events("spawn")
    assert _arg(spawns[1]["argv"], "--resume-session-at") == UUID_A
    assert "--session-id" in spawns[2]["argv"]
    assert fake_cli.events("turn")[-1]["content"] == "FULL PROMPT"


def _rewound_retry(session, cli):
    """Turn A, turn B (+ a tool loop), then /retry of turn B."""

    turn_a = [{"role": "system", "content": "rules"}, {"role": "user", "content": "u1"}]
    _run(session, cli, turn_a)
    turn_b = turn_a + [{"role": "assistant", "content": "F1"}, {"role": "user", "content": "u2"}]
    _run(session, cli, turn_b)
    _run(session, cli, _tool_turn(turn_b))
    _run(session, cli, turn_b)
    return turn_b


def test_rewind_needs_resume_session_at(fake_cli, monkeypatch):
    """Regression: without the flag a plain --resume loads the abandoned
    branch, so sending only the rewound tail showed Claude both branches."""

    import agent.claude_code_session as ccs

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    monkeypatch.setattr(ccs, "_resume_at_supported", False)
    _rewound_retry(ClaudeCodeSession(), fake_cli)
    turn = fake_cli.events("turn")[-1]
    assert turn["content"] == "FULL PROMPT"
    assert turn["context"] == ["FULL PROMPT"]


def test_rewind_then_advance_stays_on_the_new_branch(fake_cli, monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    session = ClaudeCodeSession()
    turn_b = _rewound_retry(session, fake_cli)
    turn_c = turn_b + [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "u3"}]
    _run(session, fake_cli, turn_c)
    context = fake_cli.events("turn")[-1]["context"]
    assert not any("Tool Result" in entry for entry in context)
    assert sum(entry == "User:\nu2" for entry in context) == 1
    assert context[-1] == "User:\nu3"


def test_cli_rejecting_resume_session_at_falls_back_then_resumes_plainly(fake_cli, monkeypatch):
    import agent.claude_code_session as ccs

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    monkeypatch.setenv("FAKE_CLAUDE_REJECT_RESUME_AT", "1")
    monkeypatch.setattr(ccs, "_resume_at_supported", True)
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "more"}]
    assert _run(session, fake_cli, history)[0] == "Fake reply."
    assert ccs._resume_at_supported is False
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "third"}]
    _run(session, fake_cli, history)
    spawns = fake_cli.events("spawn")
    assert "--resume-session-at" in spawns[1]["argv"]  # rejected
    assert "--session-id" in spawns[2]["argv"]  # fresh fallback
    assert "--resume" in spawns[3]["argv"] and "--resume-session-at" not in spawns[3]["argv"]
    assert fake_cli.events("turn")[-1]["content"] == "User:\nthird"


def test_rate_limit_error_retry_resumes_at_the_checkpoint(fake_cli, monkeypatch):
    """An is_error 429 on the warm process is retried cold at the checkpoint."""

    monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
    fake_cli.script({}, {"rate_limit": True}, {"text": "The listing shows one file."})
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    response, _ = _run(session, fake_cli, _tool_turn(history))
    assert response == "The listing shows one file."
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2 and "--resume-session-at" in spawns[1]["argv"]
    context = fake_cli.events("turn")[-1]["context"]
    assert sum("Tool Result" in entry for entry in context) == 1


def test_aborted_turn_is_dropped_on_retry(fake_cli):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    fake_cli.script({"sleep": 5})
    timer = threading.Timer(0.5, session.abort)
    timer.start()
    with pytest.raises(RuntimeError, match="aborted"):
        _run(session, fake_cli, _tool_turn(history))
    timer.join()
    fake_cli.script({})
    _run(session, fake_cli, _tool_turn(history))
    context = fake_cli.events("turn")[-1]["context"]
    assert sum("Tool Result" in entry for entry in context) == 1


def test_stateless_retry_without_warm_process_goes_fresh_directly(fake_cli, monkeypatch):
    """Regression: a soft-limit retry of a stateless (review-fork) session
    spawned ``--no-session-persistence --resume``, which can never work."""

    monkeypatch.setattr("agent.claude_code_session.time.sleep", lambda *_a, **_k: None)
    banner = "You've hit your monthly spend limit · raise it at claude.ai/settings/usage"
    fake_cli.script({}, {"text": banner}, {"text": "Saved the preference to memory."})
    fork = ClaudeCodeSession()
    review = [{"role": "user", "content": "Review the conversation above"}]
    _run(fork, fake_cli, review, state_key=None, keepalive=True)
    review += [{"role": "assistant", "content": "", "tool_calls": [_tool_call("m1", "memory")]},
               {"role": "tool", "tool_call_id": "m1", "content": "saved"}]
    response, _ = _run(fork, fake_cli, review, state_key=None, keepalive=True)
    assert response == "Saved the preference to memory."
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2
    assert not any("--resume" in spawn["argv"] for spawn in spawns)
    assert fake_cli.events("turn")[-1]["content"] == "FULL PROMPT"
    fork.shutdown()


# ---------------------------------------------------------------------------
# Identity, diagnostics, retention
# ---------------------------------------------------------------------------


def test_changed_system_prompt_starts_a_fresh_session(fake_cli, caplog):
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history, system_prompt="RULES v1")
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "more"}]
    with caplog.at_level(logging.INFO, logger="agent.claude_code_session"):
        _run(session, fake_cli, history, system_prompt="RULES v2")
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 2 and "--session-id" in spawns[1]["argv"]
    assert "changed=system_prompt" in caplog.text


def test_resume_skip_logs_the_first_divergence(fake_cli, caplog):
    session = ClaudeCodeSession()
    _run(session, fake_cli, [{"role": "system", "content": "rules"}, {"role": "user", "content": "u1"}])
    edited = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "u1 (edited)"},
        {"role": "assistant", "content": "Fake reply."},
        {"role": "user", "content": "u2"},
    ]
    with caplog.at_level(logging.WARNING, logger="agent.claude_code_session"):
        _run(session, fake_cli, edited)
    assert "resume skipped (prefix)" in caplog.text
    assert "first_divergence=index=1 role=user previous_role=user" in caplog.text


def test_prune_removes_only_idle_unlocked_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fingerprints = _message_fingerprint([{"role": "user", "content": "hi"}])
    for key in ("idle", "busy", "active"):
        _save_durable_state(key, SID, fingerprints, model="sonnet", effort=None, tools_digest="")
        _state_path(key).with_suffix(".lock").touch()
    old = time.time() - 40 * 86400
    for key in ("idle", "busy"):
        for path in (_state_path(key), _state_path(key).with_suffix(".lock")):
            os.utime(path, (old, old))
    orphan = _state_dir() / ("f" * 64 + ".lock")
    orphan.touch()
    os.utime(orphan, (old, old))

    held = os.open(_state_path("busy").with_suffix(".lock"), os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        assert _prune_durable_states() == 1
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)
    assert not _state_path("idle").exists()
    assert not _state_path("idle").with_suffix(".lock").exists()
    assert not orphan.exists()
    assert _state_path("busy").exists(), "a key in use is never pruned"
    assert _state_path("active").exists()
