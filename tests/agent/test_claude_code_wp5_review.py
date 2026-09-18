"""Claude Code bridge WP5 review: adversarial regression tests.

Plan-window warnings that share a label, stale warnings and plan figures past
a window's reset, fallback notices without a named model, control requests
racing turns on the warm process.
"""

from __future__ import annotations

import time

import pytest

import agent.claude_code_session as ccs
from agent.claude_code_session import ClaudeCodeSession
from tests.agent.test_claude_code_wp5 import (  # noqa: F401 - fake_cli is a fixture
    _history,
    _run,
    fake_cli,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    yield
    ccs.clear_usage_limit_blocks()


def _weekly_twins(utilization, reset):
    # Claude Code lists the weekly window twice when extra usage is off:
    # ``seven_day`` and ``seven_day_overage_included`` carry the same numbers.
    return {
        "status": "allowed",
        "rateLimitType": "seven_day",
        "resetsAt": reset,
        "unifiedWindows": {
            "five_hour": {"utilization": 0.2, "resetsAt": reset - 86400},
            "seven_day": {"utilization": utilization, "resetsAt": reset},
            "seven_day_overage_included": {"utilization": utilization, "resetsAt": reset},
        },
    }


def test_weekly_twin_windows_are_announced_once_per_cycle():
    reset = int(time.time()) + 3 * 86400
    session = ClaudeCodeSession()
    session._record_rate_limit(_weekly_twins(0.95, reset))
    first = session.take_notices()
    assert len(first) == 1 and "weekly limit is 95% used" in first[0]
    # The next event of the same cycle (a later turn, another process).
    session._record_rate_limit(_weekly_twins(0.96, reset))
    assert session.take_notices() == []
    other = ClaudeCodeSession()
    other._record_rate_limit(_weekly_twins(0.97, reset))
    assert other.take_notices() == []


def test_twin_warning_seen_by_another_session_is_not_repeated():
    reset = int(time.time()) + 3 * 86400
    first = ClaudeCodeSession()
    only_week = _weekly_twins(0.95, reset)
    del only_week["unifiedWindows"]["seven_day_overage_included"]
    first._record_rate_limit(only_week)
    assert len(first.take_notices()) == 1
    # The twin shows up hot later in the same cycle: same limit, no repeat.
    later = ClaudeCodeSession()
    later._record_rate_limit(_weekly_twins(0.97, reset))
    assert later.take_notices() == []


def test_pending_warning_carries_the_latest_figures():
    reset = int(time.time()) + 3600
    session = ClaudeCodeSession()
    info = {
        "status": "allowed",
        "unifiedWindows": {"five_hour": {"utilization": 0.91, "resetsAt": reset}},
    }
    session._record_rate_limit(info)
    info["unifiedWindows"]["five_hour"]["utilization"] = 0.97
    session._record_rate_limit(info)
    notices = session.take_notices()
    assert len(notices) == 1 and "97% used" in notices[0]


def test_pending_warning_whose_window_reset_is_dropped(monkeypatch):
    reset = time.time() + 3600
    session = ClaudeCodeSession()
    session._record_rate_limit(
        {"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.95, "resetsAt": reset}}}
    )
    # No reply carried it before the window reset (hours without a turn).
    real_time = time.time
    monkeypatch.setattr(ccs.time, "time", lambda: real_time() + 7200)
    assert session.take_notices() == []


def test_fallback_notice_without_a_named_fallback_model():
    text = ccs._model_fallback_notice(
        "model_refusal_fallback", {"originalModel": "claude-opus-5", "apiRefusalCategory": "cyber"}
    )
    assert "None" not in text and "another model answered instead" in text
    assert "None" not in ccs._model_fallback_notice("model_fallback", {})


def _stale_info(now):
    # Reported at 09:00: the 5-hour window reset since, the week has not.
    return {
        "status": "rejected",
        "rateLimitType": "five_hour",
        "resetsAt": now - 600,
        "unifiedWindows": {
            "five_hour": {"utilization": 1.0, "resetsAt": now - 600},
            "seven_day": {"utilization": 0.41, "resetsAt": now + 3 * 86400},
        },
    }


def test_plan_line_and_footer_skip_windows_that_reset_since_the_report():
    from types import SimpleNamespace

    from gateway import claude_code_commands as cc

    now = time.time()
    session = ClaudeCodeSession()
    session._record_rate_limit(_stale_info(now))
    line = cc.plan_line(session)
    assert "week 41%" in line
    assert "5h" not in line and "limit reached" not in line
    agent = SimpleNamespace(provider="claude-code", client=SimpleNamespace(_claude_session=session))
    meta = cc.footer_meta(agent)
    assert "plan_5h" not in meta and meta["plan_7d"] == pytest.approx(0.41)


def test_usage_fallback_marks_windows_that_reset_since_the_report():
    import agent.account_usage as account_usage

    now = time.time()
    snapshot = account_usage.build_claude_code_usage_snapshot(
        rate_limit=_stale_info(now), recorded_at=now - 3 * 3600, note="no answer"
    )
    text = "\n".join(account_usage.render_account_usage_lines(snapshot))
    assert "Current week: 59% remaining (41% used)" in text
    assert "Current session (5h): reset at" in text and "after this report" in text
    assert "0% remaining (100% used)" not in text
    assert "Status: rejected" not in text


def _slow_control_cli(fake_cli, delay):
    from tests.agent.test_claude_code_wp5 import FAKE_CLI

    marker = '    if ignore_control:\n        return\n'
    assert marker in FAKE_CLI
    slowed = FAKE_CLI.replace(marker, marker + f"    time.sleep({delay})\n", 1)
    (fake_cli.tmp / "fake_claude.py").write_text(slowed)


def test_turn_waits_for_a_control_request_on_the_warm_process(fake_cli):
    import threading

    from tests.agent.test_claude_code_wp5 import _control

    _slow_control_cli(fake_cli, 0.8)
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    answers = {}

    def ask():
        answers["models"] = _control(session, fake_cli, [("list_models", {})])

    asker = threading.Thread(target=ask)
    asker.start()
    deadline = time.time() + 5
    while not fake_cli.events("control") and time.time() < deadline:
        time.sleep(0.02)
    # The agent's next model call arrives while the warm process answers.
    response, _reasoning = _run(session, fake_cli, _history(2))
    asker.join(10)
    assert response == "Fake reply."
    assert answers["models"][0]["models"]
    assert len(fake_cli.events("spawn")) == 1  # both served by the warm process
    assert len(fake_cli.events("turn")) == 2
    session.shutdown()


def test_plan_snapshot_updates_hold_a_cross_process_lock():
    import fcntl
    import os

    with ccs._rate_limit_update():
        fd = os.open(ccs._state_dir() / "rate_limit.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


def _saved_main_conversation(fake_cli, monkeypatch):
    """A claude-code conversation saved under the gateway agent's key
    (``<root>|<session id>`` = ``main|main``), then a gateway restart: no
    agent is loaded for the session."""

    from tests.agent.test_claude_code_wp5 import _claude_agent

    from agent.portal_tags import reset_bridge_state_key, set_bridge_state_key

    monkeypatch.chdir(fake_cli.tmp)
    monkeypatch.setenv("HERMES_CLAUDE_CODE_COMMAND", fake_cli.command)
    agent = _claude_agent(fake_cli, _conversation_root_id=lambda: "main")
    token = set_bridge_state_key("main|main")
    try:
        agent.client.chat.completions.create(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
        )
    finally:
        reset_bridge_state_key(token)
    sid = agent.client._claude_session.conversation_ref()["session_id"]
    agent.client.close()
    return sid


def _restarted_runner():
    from tests.agent.test_claude_code_wp5 import _runner

    runner = _runner()
    runner.session_store.get_model_override.return_value = {"provider": "claude-code"}
    runner.session_store.peek_session_id.return_value = "main"
    return runner


def test_claude_reset_after_a_gateway_restart_drops_the_saved_session(fake_cli, monkeypatch):
    import asyncio

    from tests.agent.test_claude_code_wp5 import _event

    sid = _saved_main_conversation(fake_cli, monkeypatch)
    assert ccs._state_path("main|main").exists()
    runner = _restarted_runner()
    text = asyncio.run(runner._handle_claude_command(_event("reset")))
    assert f"`{sid[:8]}`" in text
    assert not ccs._state_path("main|main").exists()


def test_claude_handoff_and_context_after_a_gateway_restart(fake_cli, monkeypatch):
    import asyncio

    from tests.agent.test_claude_code_wp5 import _event

    sid = _saved_main_conversation(fake_cli, monkeypatch)
    runner = _restarted_runner()
    handoff = asyncio.run(runner._handle_claude_command(_event("handoff")))
    assert f"claude --resume {sid} --fork-session" in handoff
    context = asyncio.run(runner._handle_claude_command(_event("context")))
    assert "Claude Code: 1,200 / 200,000 tokens" in context
    status = asyncio.run(runner._handle_claude_command(_event("status")))
    assert f"`{sid[:8]}` (saved; loaded on the next message)" in status


TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]


def _hot(reset):
    return {
        "status": "allowed",
        "unifiedWindows": {"five_hour": {"utilization": 0.94, "resetsAt": reset}},
    }


def test_plan_warning_reaching_an_agent_without_a_chat_surface_is_not_lost(fake_cli):
    from types import SimpleNamespace

    from agent.claude_code_client import ClaudeCodeClient
    from agent.conversation_loop import _show_provider_notices

    reset = int(time.time()) + 3600
    fake_cli.script({"rate_limit_info": _hot(reset)})
    # A delegate subagent (or a cron job) on Claude Code sees the hot window
    # first: its status lines reach no chat.
    child_client = ClaudeCodeClient(command=fake_cli.command, cwd=str(fake_cli.tmp))
    completion = child_client.chat.completions.create(
        model="claude-opus-5", messages=[{"role": "user", "content": "hi"}], tools=TOOLS
    )
    printed = []
    child = SimpleNamespace(
        provider="claude-code", platform="subagent", status_callback=None, _emit_status=printed.append
    )
    _show_provider_notices(child, completion.usage)
    assert printed == []
    # The gateway agent's session sees the same window: still announced.
    main = ClaudeCodeSession()
    main._record_rate_limit(_hot(reset))
    assert len(main.take_notices()) == 1
    child_client.close()


def test_gateway_agent_shows_the_plan_warning_once(fake_cli):
    from types import SimpleNamespace

    from agent.claude_code_client import ClaudeCodeClient
    from agent.conversation_loop import _show_provider_notices

    reset = int(time.time()) + 3600
    fake_cli.script({"rate_limit_info": _hot(reset)}, {"rate_limit_info": _hot(reset)})
    client = ClaudeCodeClient(command=fake_cli.command, cwd=str(fake_cli.tmp))
    shown = []
    agent = SimpleNamespace(
        provider="claude-code", platform="discord", status_callback=lambda *a: None,
        _emit_status=shown.append,
    )
    for _ in range(2):
        completion = client.chat.completions.create(
            model="claude-opus-5", messages=[{"role": "user", "content": "hi"}], tools=TOOLS
        )
        _show_provider_notices(agent, completion.usage)
    assert len(shown) == 1 and "5-hour session limit is 94% used" in shown[0]
    # Another session of the same account does not repeat it.
    other = ClaudeCodeSession()
    other._record_rate_limit(_hot(reset))
    assert other.take_notices() == []
    client.close()


def test_warm_exchange_error_closes_the_process_and_falls_back(fake_cli, monkeypatch):
    from tests.agent.test_claude_code_wp5 import _control

    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    warm = session._warm
    real = ccs._exchange_control
    calls = []

    def flaky(process, requests, **kwargs):
        calls.append(process)
        if process is warm.process:
            raise OSError("pipe trouble")
        return real(process, requests, **kwargs)

    monkeypatch.setattr(ccs, "_exchange_control", flaky)
    (models,) = _control(session, fake_cli, [("list_models", {})])
    assert models["models"]
    # The taken warm process is not left running unowned.
    assert session._warm is None and not warm.alive()
    assert len(calls) == 2
    session.shutdown()


def test_no_control_requests_start_no_process(fake_cli):
    from tests.agent.test_claude_code_wp5 import _control

    assert _control(ClaudeCodeSession(), fake_cli, []) == []
    assert ccs.run_control_requests([], command=fake_cli.command) == []
    assert fake_cli.events("spawn") == []


def test_reset_does_not_wait_forever_behind_a_stuck_request(fake_cli, monkeypatch):
    import threading
    from types import SimpleNamespace

    from gateway import claude_code_commands as cc

    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    held = threading.Event()
    release = threading.Event()

    def hold():
        with session._lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    monkeypatch.setattr(cc, "_RESET_WAIT_SECONDS", 0.2)
    agent = SimpleNamespace(provider="claude-code", client=SimpleNamespace(_claude_session=session))
    try:
        started = time.monotonic()
        text = cc.do_reset(agent)
        assert time.monotonic() - started < 5
        assert "still finishing a request" in text
        assert ccs._state_path("root|main").exists()  # nothing dropped
    finally:
        release.set()
        holder.join(5)
    assert session.reset_conversation(wait=1.0)
    assert not ccs._state_path("root|main").exists()
    session.shutdown()


def test_repeated_fallback_notice_is_not_shown_on_every_call(fake_cli, monkeypatch):
    refusal = {
        "system": [
            {
                "subtype": "model_refusal_fallback",
                "originalModel": "claude-opus-5",
                "fallbackModel": "claude-opus-4-8",
                "scope": "session",
                "content": "Opus 5's safeguards flagged this message. Switched to Opus 4.8.",
            }
        ],
        "model": "claude-opus-4-8",
    }
    # A flagged conversation: every call of the tool loop resumes on Opus 5
    # (a session-wide fallback is never parked) and falls back again.
    fake_cli.script(refusal, refusal, refusal)
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    assert len(session.take_notices()) == 1
    _run(session, fake_cli, _history(2))
    assert session.take_notices() == []
    # Much later it is news again.
    real_time = time.time
    monkeypatch.setattr(ccs.time, "time", lambda: real_time() + 3600)
    _run(session, fake_cli, _history(3))
    assert len(session.take_notices()) == 1
    session.shutdown()
