"""Adversarial regression tests from the WP3 review (streaming, interrupts)."""

from __future__ import annotations

import threading
import time

import pytest

import agent.claude_code_session as ccs
from agent.claude_code_session import ClaudeCodeInterrupted, ClaudeCodeSession
from tests.agent.test_claude_code_wp3 import (  # noqa: F401 - fake_cli is a fixture
    LONG,
    _interrupt_second_turn,
    _run,
    _wait_for,
    fake_cli,
)


def test_interrupt_escalation_never_kills_the_next_turn(fake_cli, monkeypatch):
    """The 1.5 s escalation timer of a graceful interrupt that already ended
    must not kill the next request that reuses the same warm process."""

    monkeypatch.setattr(ccs, "_INTERRUPT_GRACE_SECONDS", 0.6)
    fake_cli.script(
        {"text": "Hello."},
        {"text": LONG, "delay": 0.02, "chunk": 5},
        {"text": "Slow next answer. " * 20, "delay": 0.02, "chunk": 5},
    )
    session = ClaudeCodeSession()
    history = [{"role": "user", "content": "hi"}]
    _run(session, fake_cli, history)
    history += [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "count"}]
    exc, _elapsed = _interrupt_second_turn(session, fake_cli, history)
    assert isinstance(exc, ClaudeCodeInterrupted)
    history += [{"role": "user", "content": "do this instead"}]
    response, _ = _run(session, fake_cli, history)
    assert response.startswith("Slow next answer.")
    assert not fake_cli.events("sigterm")
    assert len(fake_cli.events("spawn")) == 1
    session.shutdown()


def test_a_reply_discussing_the_flag_does_not_turn_it_off(monkeypatch):
    """Only the CLI's own argument error (stderr) disables --thinking-display;
    a crashed turn whose streamed reply quotes such an error must not."""

    import json

    monkeypatch.setattr(ccs, "_thinking_display_supported", True)
    monkeypatch.setattr(ccs, "_resume_at_supported", True)
    session = ClaudeCodeSession()
    quoted = json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": (
            "The CLI printed: error: unknown option '--thinking-display'. "
            "Allowed choices are summarized, omitted. Same for '--resume-session-at'."
        )}]},
    })
    with pytest.raises(RuntimeError) as info:
        session._raise_process_failure(
            137, quoted, "", "12345678-1234-1234-1234-123456789abc"
        )
    assert type(info.value) is RuntimeError
    assert ccs._thinking_display() == "summarized"
    assert ccs._resume_at_supported is True


def test_interrupted_rewind_continues_without_the_abandoned_branch(fake_cli):
    """An interrupted rewind (/retry) continues right after the interruption
    instead of replaying the whole transcript, and never keeps checkpoints of
    the branch it abandoned: a later rewind would otherwise resume in the old
    branch while Hermes's transcript holds the new one."""

    fake_cli.script(
        {"text": "a1"}, {"text": "a2"}, {"text": "a3"},
        {"text": LONG, "delay": 0.02, "chunk": 5},  # the rewound turn, interrupted
        {"text": "a5"}, {"text": "a6"},
    )
    session = ClaudeCodeSession()
    h = [{"role": "user", "content": "u1"}]
    _run(session, fake_cli, h)
    h += [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    _run(session, fake_cli, h)
    old_branch = session._checkpoints[-1][1]  # Claude's reply to u2
    h += [{"role": "assistant", "content": "a2"}, {"role": "user", "content": "u3"}]
    _run(session, fake_cli, h)
    # /retry with an edited second message: rewinds to the first reply.
    branch = h[:2] + [{"role": "user", "content": "u2-edited"}]
    exc, _elapsed = _interrupt_second_turn(session, fake_cli, branch)
    assert isinstance(exc, ClaudeCodeInterrupted)
    branch += [
        {"role": "assistant", "content": "[interrupted]"},
        {"role": "user", "content": "u4"},
    ]
    assert _run(session, fake_cli, branch)[0] == "a5"
    turn = fake_cli.events("turn")[-1]
    assert "u4" in turn["content"] and "u2-edited" not in turn["content"]
    assert turn["context"][-2] == "[Request interrupted by user]"
    assert "A: a2" not in turn["context"]
    assert old_branch not in [uuid for _count, uuid in session._checkpoints]
    # /retry of the last message: rewinds within the new branch, never to a
    # checkpoint of the abandoned one (Claude Code keeps every branch).
    spawned = len(fake_cli.events("spawn"))
    rewound = branch[:4] + [{"role": "user", "content": "u5"}]
    assert _run(session, fake_cli, rewound)[0] == "a6"
    for spawn in fake_cli.events("spawn")[spawned:]:
        assert old_branch not in spawn["argv"]
    assert "A: a2" not in fake_cli.events("turn")[-1]["context"]
    session.shutdown()
