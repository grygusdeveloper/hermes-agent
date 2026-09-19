"""Claude Code bridge: Claude-only Discord features (WP5).

Plan windows (persisted ``rate_limit.json``, ``/usage``, one-time warnings),
model fallback notices, stream-json control requests without a model turn
(``get_usage``, ``get_context_usage``, ``list_models``) through the warm
process or a throwaway control-only process, ``/claude`` subcommands, the
runtime footer fields and the command registration.

Most tests drive the real ``ClaudeCodeSession``/``ClaudeCodeClient`` against a
fake stream-json CLI that answers control requests the way Claude Code 2.1.276
does (``get_usage`` returns ``rate_limits: null`` under
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1, as the real CLI was seen to).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.account_usage as account_usage
import agent.claude_code_session as ccs
from agent.claude_code_client import ClaudeCodeClient, _completion_usage
from agent.claude_code_session import (
    ClaudeCodeControlError,
    ClaudeCodeLaunchError,
    ClaudeCodeSession,
    _WARM_IDLE,
    load_rate_limit_snapshot,
    run_control_requests,
)

MODEL = "claude-opus-5"

FAKE_CLI = textwrap.dedent(
    r"""
    import json, os, sys, time, uuid
    log = os.environ["FAKE_CLAUDE_LOG"]
    store = os.environ["FAKE_CLAUDE_STORE"]
    script = os.environ.get("FAKE_CLAUDE_SCRIPT", "")
    argv = sys.argv[1:]
    MODEL = "claude-opus-5"

    def write_log(obj):
        with open(log, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    if argv[:1] == ["--version"]:
        print("9.9.9 (Claude Code)")
        sys.exit(0)
    if argv[:3] == ["auth", "status", "--json"]:
        print(json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
                          "email": "someone@example.com", "orgId": "org-1", "orgName": "Org",
                          "subscriptionType": "max"}))
        sys.exit(0)

    def arg(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    nonessential = os.environ.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
    write_log({"event": "spawn", "argv": argv, "pid": os.getpid(), "nonessential": nonessential})
    ignore_control = bool(os.environ.get("FAKE_IGNORE_CONTROL"))
    sid = arg("--resume") or arg("--session-id") or str(uuid.uuid4())
    persist = "--no-session-persistence" not in argv
    path = os.path.join(store, sid + ".json")
    chain = []
    if "--resume" in argv:
        if not os.path.exists(path):
            sys.stderr.write("No conversation found with session ID: " + sid + "\n")
            sys.exit(1)
        chain = json.load(open(path))
        if "--fork-session" in argv:
            sid = str(uuid.uuid4())

    def save():
        if persist:
            with open(os.path.join(store, sid + ".json"), "w") as fh:
                json.dump(chain, fh)

    def out(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def next_reply():
        if not script or not os.path.exists(script):
            return {}
        counter = script + ".n"
        n = int(open(counter).read()) if os.path.exists(counter) else 0
        replies = json.load(open(script))
        with open(counter, "w") as fh:
            fh.write(str(n + 1))
        return replies[n] if n < len(replies) else {}

    USAGE = {
        "session": {"total_cost_usd": 0},
        "subscription_type": "max",
        "rate_limits_available": True,
        "rate_limits": {
            "five_hour": {"utilization": 35, "resets_at": "2099-01-01T16:00:01+00:00"},
            "seven_day": {"utilization": 63, "resets_at": "2099-01-02T16:00:01+00:00"},
            "seven_day_opus": None,
            "extra_usage": {"is_enabled": False, "monthly_limit": 4900, "used_credits": 2093,
                            "currency": "EUR", "decimal_places": 2,
                            "disabled_reason": "out_of_credits"},
            "model_scoped": [{"display_name": "Fable", "utilization": 99,
                              "resets_at": "2099-01-02T16:00:00+00:00"}],
        },
        "behaviors": None,
    }

    def control(msg):
        request = msg.get("request") or {}
        subtype = request.get("subtype")
        write_log({"event": "control", "subtype": subtype, "pid": os.getpid(),
                   "request": request, "chain": len(chain)})
        if ignore_control:
            return
        if subtype == "get_usage":
            payload = dict(USAGE)
            if nonessential == "1":
                payload["rate_limits"] = None
        elif subtype == "get_context_usage":
            total = 1000 + 100 * len(chain)
            payload = {"totalTokens": total, "maxTokens": 200000, "model": MODEL,
                       "percentage": round(100 * total / 200000),
                       "categories": [
                           {"name": "System prompt", "tokens": 1000, "kind": "used"},
                           {"name": "Messages", "tokens": 100 * len(chain), "kind": "used"},
                           {"name": "Compact buffer", "tokens": 3000, "kind": "buffer"},
                           {"name": "Free space", "tokens": 196000, "kind": "free"}]}
        elif subtype == "list_models":
            payload = {"models": [
                {"value": "default", "resolvedModel": "claude-opus-5[1m]", "displayName": "Default",
                 "description": "Opus 5", "supportedEffortLevels": ["low", "high"]},
                {"value": "sonnet", "resolvedModel": "claude-sonnet-5", "displayName": "Sonnet",
                 "description": "Sonnet 5"}]}
        elif subtype == "get_binary_version":
            payload = {"version": "9.9.9"}
        else:
            out({"type": "control_response", "response": {
                "subtype": "error", "request_id": msg["request_id"],
                "error": "Unsupported control request subtype: " + str(subtype)}})
            return
        out({"type": "control_response", "response": {
            "subtype": "success", "request_id": msg["request_id"], "response": payload}})

    # Like Claude Code, --resume restores the session's cumulative cost and
    # modelUsage from the cost state its last clean exit saved (this fake
    # saves it after every turn; FAKE_NO_COST_STATE: the process was killed).
    cost_path = os.path.join(store, (arg("--resume") or sid) + ".cost.json")
    cost, tokens = 0.0, [0, 0, 0]
    if "--resume" in argv and os.path.exists(cost_path):
        saved = json.load(open(cost_path))
        cost, tokens = saved["cost"], saved["tokens"]
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("type") == "control_request":
            control(msg)
            continue
        content = msg["message"]["content"]
        chain.append({"uuid": str(uuid.uuid4()), "text": str(content)})
        reply = next_reply()
        write_log({"event": "turn", "pid": os.getpid()})
        out({"type": "system", "subtype": "init", "session_id": sid})
        for event in reply.get("system", []):
            out(dict(event, type="system", session_id=sid))
        if reply.get("rate_limit_info"):
            out({"type": "rate_limit_event", "rate_limit_info": reply["rate_limit_info"],
                 "session_id": sid})
        text = reply.get("text", "Fake reply.")
        model = reply.get("model", MODEL)
        uid = str(uuid.uuid4())
        chain.append({"uuid": uid, "text": "A: " + text})
        save()
        out({"type": "assistant", "uuid": uid, "session_id": sid,
             "message": {"id": "msg_" + uid[:8], "model": model,
                         "content": [{"type": "text", "text": text}]}})
        cost += float(reply.get("cost", 0.25))
        tokens = [tokens[0] + 100, tokens[1] + 900, tokens[2] + 50]
        out({"type": "result", "subtype": "success", "is_error": False, "result": text,
             "session_id": sid, "stop_reason": "end_turn", "total_cost_usd": cost,
             "duration_api_ms": 1200,
             "usage": {"input_tokens": 100, "cache_read_input_tokens": 900,
                       "output_tokens": 50},
             "modelUsage": {model: {"inputTokens": tokens[0], "cacheReadInputTokens": tokens[1],
                                    "outputTokens": tokens[2], "cacheCreationInputTokens": 0,
                                    "costUSD": cost}}})
        if persist and not os.environ.get("FAKE_NO_COST_STATE"):
            with open(os.path.join(store, sid + ".cost.json"), "w") as fh:
                json.dump({"cost": cost, "tokens": tokens}, fh)
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
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", raising=False)
    monkeypatch.delenv("FAKE_IGNORE_CONTROL", raising=False)
    monkeypatch.setattr(account_usage, "_claude_usage_cache", None)

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

    yield SimpleNamespace(
        command=str(launcher), events=events, script=script_replies, store=store, tmp=tmp_path
    )
    for warm in list(_WARM_IDLE.values()):
        warm.close()
    ccs.clear_usage_limit_blocks()


def _env(**extra):
    keys = ("PATH", "FAKE_CLAUDE_LOG", "FAKE_CLAUDE_STORE", "FAKE_CLAUDE_SCRIPT", "FAKE_IGNORE_CONTROL",
            "FAKE_NO_COST_STATE")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env.update(extra)
    return env


def _run(session, cli, messages, *, state_key="root|main", **kwargs):
    return session.run(
        "FULL PROMPT",
        messages=messages,
        model=MODEL,
        tools_digest="digest",
        has_tools=True,
        timeout_seconds=30,
        cwd=str(cli.tmp),
        env=kwargs.pop("env", None) or _env(),
        state_key=state_key,
        command=cli.command,
        system_prompt="SYSTEM",
        keepalive=kwargs.pop("keepalive", True),
        **kwargs,
    )


def _control(session, cli, requests, **kwargs):
    return session.control_requests(
        requests, command=cli.command, cwd=str(cli.tmp), env=_env(), **kwargs
    )


def _window_info(utilization, *, resets_at, status="allowed", kind="five_hour", weekly=0.4):
    return {
        "status": status,
        "rateLimitType": kind,
        "resetsAt": resets_at,
        "unifiedWindows": {
            "five_hour": {"utilization": utilization, "resetsAt": resets_at},
            "seven_day": {"utilization": weekly, "resetsAt": resets_at + 5 * 86400},
        },
    }


HI = [{"role": "user", "content": "hi"}]


def _history(n):
    history = [{"role": "user", "content": "hi"}]
    for index in range(n - 1):
        history += [
            {"role": "assistant", "content": "Fake reply."},
            {"role": "user", "content": f"again {index}"},
        ]
    return history


# ---------------------------------------------------------------------------
# Persisted plan snapshot and one-time warnings
# ---------------------------------------------------------------------------


def test_rate_limit_event_is_persisted_for_other_sessions(fake_cli):
    reset = int(time.time()) + 3600
    fake_cli.script({"rate_limit_info": _window_info(0.3, resets_at=reset)})
    session = ClaudeCodeSession()
    _run(session, fake_cli, HI)
    snapshot = load_rate_limit_snapshot()
    assert snapshot is not None
    assert snapshot["info"]["unifiedWindows"]["five_hour"]["utilization"] == 0.3
    assert abs(snapshot["recorded_at"] - time.time()) < 60
    path = ccs._rate_limit_path()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    # Below the warning threshold: nothing to announce.
    assert session.take_notices() == []
    session.shutdown()


def test_plan_window_warning_is_announced_once_per_window_cycle(fake_cli):
    reset = int(time.time()) + 3600
    fake_cli.script(
        {"rate_limit_info": _window_info(0.92, resets_at=reset)},
        {"rate_limit_info": _window_info(0.95, resets_at=reset)},
        # The next 5-hour window, hot again.
        {"rate_limit_info": _window_info(0.91, resets_at=reset + 5 * 3600)},
    )
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    notices = session.take_notices()
    assert len(notices) == 1
    assert "5-hour session limit is 92% used" in notices[0] and "resets" in notices[0]
    _run(session, fake_cli, _history(2))
    assert session.take_notices() == []  # same window cycle
    # A new session object (another agent, a restarted gateway) remembers.
    other = ClaudeCodeSession()
    other._record_rate_limit(_window_info(0.97, resets_at=reset))
    assert other.take_notices() == []
    _run(session, fake_cli, _history(3))
    assert [n for n in session.take_notices() if "91% used" in n]
    session.shutdown()


def test_warning_seen_first_by_an_auxiliary_call_is_not_swallowed(fake_cli):
    reset = time.time() + 3600
    hot = _window_info(0.94, resets_at=reset)
    # An auxiliary client (title, compression) records the hot window first
    # and never shows it: its calls offer no tools, so it never takes notices.
    fake_cli.script({"rate_limit_info": hot})
    aux = ClaudeCodeClient(command=fake_cli.command, cwd=str(fake_cli.tmp))
    completion = aux.chat.completions.create(model=MODEL, messages=HI)
    assert completion.usage.hermes_claim_notices is None
    # The agent's own session still announces it, once.
    main = ClaudeCodeSession()
    main._record_rate_limit(hot)
    assert len(main.take_notices()) == 1
    assert aux._claude_session.take_notices() == []  # already shown elsewhere
    other = ClaudeCodeSession()
    other._record_rate_limit(hot)
    assert other.take_notices() == []
    aux.close()


def test_cli_warning_status_is_announced_and_rejections_are_not():
    reset = time.time() + 7200
    session = ClaudeCodeSession()
    session._record_rate_limit(
        {"status": "allowed_warning", "rateLimitType": "seven_day", "resetsAt": reset}
    )
    notices = session.take_notices()
    assert len(notices) == 1 and "weekly limit is almost used up" in notices[0]
    session._record_rate_limit(_window_info(1.0, resets_at=reset + 60, status="rejected"))
    assert session.take_notices() == []


def test_expired_window_is_not_announced():
    session = ClaudeCodeSession()
    session._record_rate_limit(_window_info(0.99, resets_at=time.time() - 10))
    assert session.take_notices() == []


def test_model_refusal_fallback_becomes_a_notice(fake_cli):
    fake_cli.script(
        {
            "system": [
                {
                    "subtype": "model_refusal_fallback",
                    "originalModel": "claude-opus-5",
                    "fallbackModel": "claude-opus-4-8",
                    "scope": "session",
                    "apiRefusalCategory": "cyber",
                    "content": "Opus 5's safeguards flagged this message. Switched to Opus 4.8.",
                }
            ],
            "model": "claude-opus-4-8",
        },
        {"system": [{"subtype": "model_fallback", "originalModel": "claude-opus-5",
                     "fallbackModel": "claude-sonnet-5", "scope": "turn"}]},
    )
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    assert session.take_notices() == [
        "⚠️ Claude Code: Opus 5's safeguards flagged this message. Switched to Opus 4.8."
    ]
    assert session.last_usage["served_model"] == "claude-opus-4-8"
    _run(session, fake_cli, _history(2))
    assert session.take_notices() == [
        "⚠️ claude-opus-5 was unavailable; claude-sonnet-5 answered instead."
    ]
    session.shutdown()


def test_notices_ride_on_both_completion_paths(fake_cli):
    reset = int(time.time()) + 3600
    fake_cli.script(
        {"rate_limit_info": _window_info(0.93, resets_at=reset)},
        {"rate_limit_info": _window_info(0.5, resets_at=reset, weekly=0.96)},
    )
    client = ClaudeCodeClient(command=fake_cli.command, cwd=str(fake_cli.tmp))
    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
    completion = client.chat.completions.create(model=MODEL, messages=_history(1), tools=tools)
    notices = completion.usage.hermes_claim_notices()
    assert len(notices) == 1
    assert "5-hour session limit is 93% used" in notices[0]
    chunks = list(
        client.chat.completions.create(model=MODEL, messages=_history(2), tools=tools, stream=True)
    )
    usage = [chunk.usage for chunk in chunks if getattr(chunk, "usage", None)][-1]
    notices = usage.hermes_claim_notices()
    assert len(notices) == 1 and "weekly limit is 96% used" in notices[0]
    client.close()


def test_completion_usage_defaults_to_no_notices():
    assert _completion_usage({}).hermes_claim_notices is None


def test_conversation_loop_shows_bridge_notices_only_for_claude_code():
    from agent.conversation_loop import _show_provider_notices

    shown = []
    claims = []

    def claim():
        claims.append(1)
        return ["⚠️ Claude weekly limit is 95% used.", "", 3]

    agent = SimpleNamespace(
        provider="claude-code", status_callback=lambda *a: None, _emit_status=shown.append
    )
    usage = SimpleNamespace(hermes_claim_notices=claim)
    _show_provider_notices(agent, usage)
    assert shown == ["⚠️ Claude weekly limit is 95% used."]
    other = SimpleNamespace(
        provider="openrouter", status_callback=lambda *a: None, _emit_status=shown.append
    )
    _show_provider_notices(other, usage)
    _show_provider_notices(agent, SimpleNamespace())
    # An agent whose status lines reach no one (a subagent) never claims.
    unseen = SimpleNamespace(provider="claude-code", platform="subagent", _emit_status=shown.append)
    _show_provider_notices(unseen, usage)
    assert len(shown) == 1 and len(claims) == 1
    cli = SimpleNamespace(provider="claude-code", platform="cli", _emit_status=shown.append)
    _show_provider_notices(cli, usage)
    assert len(shown) == 2


@pytest.mark.parametrize("keepalive", ["0", "30"])
@pytest.mark.parametrize("cost_state", [True, False])
def test_turn_cost_is_the_turns_own_on_a_resumed_process(fake_cli, monkeypatch, keepalive, cost_state):
    """Regression (f6): ``--resume`` restores the session's cumulative cost,
    so every turn served by a new process (warm expiry, gateway restart)
    reported the whole session's cost: footers showed $1, $2, $3, $4."""

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", keepalive)
    if not cost_state:
        monkeypatch.setenv("FAKE_NO_COST_STATE", "1")  # killed: nothing restored
    fake_cli.script(*[{"cost": 1.0}] * 4)
    session = ClaudeCodeSession()
    seen = []
    for n in range(1, 5):
        before = session.cost_total_usd
        _run(session, fake_cli, _history(n))
        seen.append((session.last_usage["total_cost_usd"], session.cost_total_usd - before))
    assert seen == [(pytest.approx(1.0), pytest.approx(1.0))] * 4
    spawns = fake_cli.events("spawn")
    assert len(spawns) == (4 if keepalive == "0" else 1)
    session.shutdown()


def test_turn_cost_survives_a_gateway_restart(fake_cli, monkeypatch):
    """A new ClaudeCodeSession (restart, agent-cache eviction) knows the
    session's last total from the durable state."""

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    fake_cli.script(*[{"cost": 1.0}] * 3)
    _run(ClaudeCodeSession(), fake_cli, _history(1))
    _run(ClaudeCodeSession(), fake_cli, _history(2))
    assert ccs._load_durable_state("root|main").cli_cost_usd == pytest.approx(2.0)
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(3))
    assert "--resume" in fake_cli.events("spawn")[-1]["argv"]
    assert session.last_usage["total_cost_usd"] == pytest.approx(1.0)
    assert session.cost_total_usd == pytest.approx(1.0)


def test_unknown_restored_cost_is_left_out(fake_cli, monkeypatch):
    """No known baseline for a restored session: no cost rather than the
    session's total (the footer then leaves the cost out)."""

    from gateway import claude_code_commands as cc

    monkeypatch.setenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", "0")
    fake_cli.script(*[{"cost": 1.0}] * 2)
    _run(ClaudeCodeSession(), fake_cli, _history(1))
    state = ccs._load_durable_state("root|main")
    ccs._save_durable_state(  # a state written before the baseline was kept
        "root|main", state.session_id, state.fingerprints, model=state.model,
        effort=state.effort, tools_digest=state.tools_digest,
        system_digest=state.system_digest, checkpoints=state.checkpoints,
    )
    agent = SimpleNamespace(provider="claude-code", model=MODEL,
                            client=SimpleNamespace(_claude_session=ClaudeCodeSession()))
    before = cc.cost_total(agent)
    _run(agent.client._claude_session, fake_cli, _history(2))
    assert agent.client._claude_session.last_usage["total_cost_usd"] is None
    assert "turn_cost_usd" not in (cc.footer_meta(agent, cost_before=before) or {})


def test_cost_total_sums_every_cli_turn(fake_cli):
    fake_cli.script({"cost": 0.25}, {"cost": 0.5})
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    _run(session, fake_cli, _history(2))
    assert len(fake_cli.events("spawn")) == 1  # one warm process, cumulative cost
    assert session.cost_total_usd == pytest.approx(0.75)
    session.shutdown()


# ---------------------------------------------------------------------------
# Control requests
# ---------------------------------------------------------------------------


def test_control_requests_without_a_session_use_a_throwaway_process(fake_cli):
    session = ClaudeCodeSession()
    models, version = _control(session, fake_cli, [("list_models", {}), ("get_binary_version", {})])
    assert [item["value"] for item in models["models"]] == ["default", "sonnet"]
    assert version == {"version": "9.9.9"}
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 1
    argv = spawns[0]["argv"]
    assert "--no-session-persistence" in argv and "--resume" not in argv
    assert fake_cli.events("turn") == []  # never a model turn


def test_get_usage_runs_without_the_nonessential_traffic_default(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, HI)
    (usage,) = _control(session, fake_cli, [("get_usage", {"skip_behaviors": True})])
    assert usage["rate_limits"]["five_hour"]["utilization"] == 35
    spawns = fake_cli.events("spawn")
    # The model turn's process keeps the default; get_usage never goes to it.
    assert spawns[0]["nonessential"] == "1"
    assert len(spawns) == 2 and spawns[1]["nonessential"] is None
    controls = fake_cli.events("control")
    assert controls[0]["pid"] == spawns[1]["pid"]
    assert controls[0]["request"] == {"subtype": "get_usage", "skip_behaviors": True}
    session.shutdown()


def test_warm_process_answers_context_and_stays_parked(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    (context,) = _control(
        session, fake_cli, [("get_context_usage", {"detail": "summary"})], conversation=True
    )
    assert context["totalTokens"] == 1200  # the warm process's own conversation
    spawns = fake_cli.events("spawn")
    assert len(spawns) == 1
    assert fake_cli.events("control")[0]["pid"] == spawns[0]["pid"]
    # The exchange left the process usable: the next turn reuses it.
    response, _ = _run(session, fake_cli, _history(2))
    assert response == "Fake reply."
    assert len(fake_cli.events("spawn")) == 1
    session.shutdown()


def test_context_of_a_saved_conversation_is_read_without_writing(fake_cli):
    first = ClaudeCodeSession()
    _run(first, fake_cli, _history(1), keepalive=False)
    sid = first.conversation_ref()["session_id"]
    before = (fake_cli.store / f"{sid}.json").read_text()
    files = sorted(os.listdir(fake_cli.store))
    # A new agent (restart): nothing bound, only the durable state.
    fresh = ClaudeCodeSession()
    (context,) = _control(
        fresh,
        fake_cli,
        [("get_context_usage", {"detail": "summary"})],
        conversation=True,
        state_key="root|main",
    )
    assert context["totalTokens"] == 1200
    argv = fake_cli.events("spawn")[-1]["argv"]
    assert argv[argv.index("--resume") + 1] == sid
    assert "--fork-session" in argv and "--no-session-persistence" in argv
    assert argv[argv.index("--model") + 1] == MODEL
    assert (fake_cli.store / f"{sid}.json").read_text() == before
    assert sorted(os.listdir(fake_cli.store)) == files
    assert fake_cli.events("turn")[-1:] and len(fake_cli.events("turn")) == 1


def test_conversation_request_without_a_saved_conversation_fails_cleanly(fake_cli):
    session = ClaudeCodeSession()
    (answer,) = _control(
        session, fake_cli, [("get_context_usage", {})], conversation=True, state_key="nothing|here"
    )
    assert isinstance(answer, ClaudeCodeControlError)
    assert fake_cli.events("spawn") == []


def test_busy_session_answers_from_a_throwaway_without_waiting(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    held = threading.Event()
    release = threading.Event()

    def hold():
        with session._lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(5)
    try:
        started = time.monotonic()
        (models,) = _control(session, fake_cli, [("list_models", {})])
        assert time.monotonic() - started < 8
        assert models["models"]
        spawns = fake_cli.events("spawn")
        assert len(spawns) == 2
        assert fake_cli.events("control")[0]["pid"] == spawns[1]["pid"]
    finally:
        release.set()
        thread.join(5)
    session.shutdown()


def test_silent_warm_process_is_closed_and_the_throwaway_answers(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_IGNORE_CONTROL", "1")
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    warm = session._warm
    monkeypatch.setattr(ccs, "_WARM_CONTROL_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    (answer,) = session.control_requests(
        [("list_models", {})], command=fake_cli.command, cwd=str(fake_cli.tmp),
        env=_env(), timeout=1.0,
    )
    assert time.monotonic() - started < 10
    assert isinstance(answer, ClaudeCodeControlError)
    assert session._warm is None and not warm.alive()
    # The warm attempt, then one throwaway (which ignored it too).
    assert len(fake_cli.events("spawn")) == 2
    session.shutdown()


def test_unknown_control_request_is_an_error_item(fake_cli):
    (answer,) = run_control_requests(
        [("get_nothing", {})], command=fake_cli.command, cwd=str(fake_cli.tmp), env=_env()
    )
    assert isinstance(answer, ClaudeCodeControlError)
    assert "Unsupported control request subtype" in str(answer)


def test_control_process_that_cannot_start_raises_a_launch_error(tmp_path):
    with pytest.raises(ClaudeCodeLaunchError):
        run_control_requests([("list_models", {})], command=str(tmp_path / "missing-claude"))


# ---------------------------------------------------------------------------
# Reset, describe, diagnostics
# ---------------------------------------------------------------------------


def test_reset_conversation_drops_only_the_claude_side(fake_cli):
    session = ClaudeCodeSession()
    history = _history(1)
    _run(session, fake_cli, history)
    sid = session.conversation_ref()["session_id"]
    assert ccs._state_path("root|main").exists()
    ccs._block_usage_limit(MODEL, {}, "You've hit your limit · resets 3pm")
    assert session.reset_conversation() == sid
    assert not ccs._state_path("root|main").exists()
    assert session.conversation_ref() is None and session._warm is None
    assert ccs._usage_limit_block(MODEL) is None
    # The same Hermes history now starts a fresh Claude session.
    history += [{"role": "assistant", "content": "Fake reply."}, {"role": "user", "content": "more"}]
    _run(session, fake_cli, history)
    argv = fake_cli.events("spawn")[-1]["argv"]
    assert "--resume" not in argv and "--session-id" in argv
    session.shutdown()


def test_describe_reports_the_session_without_prompt_text(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, _history(1))
    info = session.describe()
    assert info["session_id"] and info["persisted"] is True
    assert info["model"] == MODEL and info["messages"] == 1
    assert info["warm"]["parked"] is True and info["busy"] is False
    assert info["last_usage"]["cached_tokens"] == 900
    assert "hi" not in json.dumps(info)
    session.shutdown()


def test_cli_diagnostics_keep_only_non_secret_auth_fields(fake_cli):
    facts = ccs.claude_cli_diagnostics(fake_cli.command, env=_env())
    assert facts["version"] == "9.9.9 (Claude Code)"
    assert facts["auth"] == {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
    assert facts["path"] == os.path.realpath(fake_cli.command)
    assert facts["errors"] == []


def test_cli_diagnostics_report_a_missing_binary(tmp_path):
    facts = ccs.claude_cli_diagnostics(str(tmp_path / "nope"))
    assert facts["path"] is None and facts["errors"]


def test_bridge_diagnostics_count_states_but_not_the_plan_snapshot(fake_cli):
    session = ClaudeCodeSession()
    _run(session, fake_cli, HI)
    session._record_rate_limit(_window_info(0.1, resets_at=time.time() + 60))
    facts = ccs.bridge_diagnostics()
    assert facts["state_files"] == 1
    assert facts["warm_parked"] == 1
    assert set(facts["flags"]) == {"--thinking-display", "--thinking", "--resume-session-at"}
    session.shutdown()


# ---------------------------------------------------------------------------
# /usage: account snapshot
# ---------------------------------------------------------------------------

GET_USAGE = {
    "subscription_type": "max",
    "rate_limits_available": True,
    "rate_limits": {
        "five_hour": {"utilization": 35, "resets_at": "2099-01-01T16:00:01+00:00"},
        "seven_day": {"utilization": 63, "resets_at": "2099-01-02T16:00:01+00:00"},
        "seven_day_opus": None,
        "extra_usage": {
            "is_enabled": False,
            "monthly_limit": 4900,
            "used_credits": 2093,
            "currency": "EUR",
            "decimal_places": 2,
            "disabled_reason": "out_of_credits",
        },
        "model_scoped": [
            {"display_name": "Fable", "utilization": 99, "resets_at": "2099-01-02T16:00:00+00:00"}
        ],
    },
}


def test_get_usage_snapshot_renders_plan_windows():
    snapshot = account_usage.build_claude_code_usage_snapshot(GET_USAGE)
    lines = account_usage.render_account_usage_lines(snapshot, markdown=True)
    text = "\n".join(lines)
    assert "Provider: claude-code (Max)" in text
    assert "Current session (5h): 65% remaining (35% used)" in text
    assert "Current week: 37% remaining (63% used)" in text
    assert "Fable week: 1% remaining (99% used)" in text
    assert "resets in" in text
    assert "Extra usage: off (out of credits) · 20.93 / 49.00 EUR used" in text


def test_get_usage_snapshot_reads_scoped_limits_rows():
    payload = json.loads(json.dumps(GET_USAGE))
    payload["rate_limits"]["model_scoped"] = None
    payload["rate_limits"]["limits"] = [
        {"kind": "session", "percent": 35},
        {"kind": "weekly_scoped", "percent": 100, "severity": "critical",
         "resets_at": "2099-01-02T16:00:00+00:00",
         "scope": {"model": {"id": None, "display_name": "Fable"}}},
    ]
    snapshot = account_usage.build_claude_code_usage_snapshot(payload)
    labels = {window.label: window.used_percent for window in snapshot.windows}
    assert labels["Fable week"] == 100.0


def test_rate_limit_snapshot_renders_fractions_status_and_age():
    reset = time.time() + 3600
    info = _window_info(0.28, resets_at=reset, status="allowed_warning", weekly=0.41)
    info["overageStatus"] = "rejected"
    info["overageDisabledReason"] = "out_of_credits"
    snapshot = account_usage.build_claude_code_usage_snapshot(
        rate_limit=info, recorded_at=time.time() - 120, note="get_usage timed out"
    )
    text = "\n".join(account_usage.render_account_usage_lines(snapshot))
    assert "Current session (5h): 72% remaining (28% used)" in text
    assert "Current week: 59% remaining (41% used)" in text
    assert "Status: allowed warning (5-hour session limit)" in text
    assert "Extra usage: rejected (out of credits)" in text
    assert "As reported by Claude Code at" in text
    assert "Live numbers unavailable: get_usage timed out" in text


def test_fetch_falls_back_to_the_persisted_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(account_usage, "_claude_usage_cache", None)
    ClaudeCodeSession()._record_rate_limit(_window_info(0.5, resets_at=time.time() + 3600))
    client = SimpleNamespace(
        control_requests=lambda requests, timeout=None: [ClaudeCodeControlError("no answer")],
        _claude_session=None,
    )
    snapshot = account_usage.fetch_claude_code_account_usage(client)
    assert snapshot.source == "claude_code_rate_limit"
    assert snapshot.windows[0].used_percent == pytest.approx(50.0)
    assert any("no answer" in detail for detail in snapshot.details)


def test_fetch_account_usage_claude_code_branch_is_cached(monkeypatch):
    calls = []

    def fake_run(requests, **kwargs):
        calls.append((requests, kwargs))
        return [GET_USAGE]

    monkeypatch.setattr(account_usage, "_claude_usage_cache", None)
    monkeypatch.setattr(ccs, "run_control_requests", fake_run)
    first = account_usage.fetch_account_usage("claude-code")
    second = account_usage.fetch_account_usage("claude-code")
    assert first is second and first.plan == "Max"
    assert len(calls) == 1
    assert calls[0][0] == [("get_usage", {"skip_behaviors": True})]


# ---------------------------------------------------------------------------
# /claude subcommands (gateway/claude_code_commands.py)
# ---------------------------------------------------------------------------


def _claude_agent(fake_cli, **overrides):
    client = ClaudeCodeClient(command=fake_cli.command, cwd=str(fake_cli.tmp))
    agent = SimpleNamespace(
        provider="claude-code",
        model=MODEL,
        client=client,
        session_id="main",
        reasoning_config={"enabled": True, "effort": "high"},
        context_compressor=SimpleNamespace(
            last_prompt_tokens=12_000, context_length=1_000_000, threshold_tokens=250_000
        ),
        _persist_disabled=False,
        _conversation_root_id=lambda: "root",
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _agent_turn(agent, fake_cli, messages):
    from agent.portal_tags import reset_bridge_state_key, set_bridge_state_key

    token = set_bridge_state_key("root|main")
    try:
        return agent.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
        )
    finally:
        reset_bridge_state_key(token)


def test_session_provider_prefers_the_primary_while_on_a_fallback():
    from gateway.claude_code_commands import session_provider

    agent = SimpleNamespace(
        provider="zai", _fallback_activated=True, _primary_runtime={"provider": "claude-code"}
    )
    assert session_provider(agent) == "claude-code"
    assert session_provider(SimpleNamespace(provider="Claude-Code")) == "claude-code"
    assert session_provider(None) == ""


def test_status_shows_session_process_last_call_and_plan(fake_cli):
    from gateway import claude_code_commands as cc

    reset = int(time.time()) + 3600
    fake_cli.script({"rate_limit_info": _window_info(0.28, resets_at=reset, weekly=0.41)})
    agent = _claude_agent(fake_cli)
    _agent_turn(agent, fake_cli, HI)
    text = cc.render_status(agent)
    sid = agent.client._claude_session.conversation_ref()["session_id"]
    assert f"`{sid[:8]}`" in text and "saved" in text
    assert "effort high" in text
    assert "warm and idle" in text
    assert "1.0k in (90% cached)" in text and "≈$0.25 API-equivalent" in text
    assert "Claude plan: 5h 28% · week 41%" in text
    agent.client.close()


def test_context_compares_hermes_and_claude_code(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli)
    _agent_turn(agent, fake_cli, HI)
    text = cc.render_context(agent)
    assert "Hermes: 12,000 / 1,000,000 tokens (1%) · compresses at 250,000" in text
    assert "Claude Code: 1,200 / 200,000 tokens (1%)" in text
    assert "• System prompt: 1,000" in text and "• Messages: 200" in text
    assert "Free space" not in text and "Compact buffer" not in text
    agent.client.close()


def test_models_marks_the_session_model(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli, model="claude-sonnet-5")
    text = cc.render_models(agent)
    assert "`sonnet` — Sonnet (`claude-sonnet-5`): Sonnet 5 ← this session" in text
    assert "effort low/high" in text
    agent.client.close()


def test_usage_uses_the_live_plan_data(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli)
    text = cc.render_usage(agent)
    assert "Current session (5h): 65% remaining (35% used)" in text
    assert "Fable week" in text
    agent.client.close()


def test_handoff_prints_the_fork_command(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli)
    assert "No Claude Code conversation is saved" in cc.render_handoff(agent)
    _agent_turn(agent, fake_cli, HI)
    sid = agent.client._claude_session.conversation_ref()["session_id"]
    text = cc.render_handoff(agent)
    assert f"cd {fake_cli.tmp} && claude --resume {sid} --fork-session" in text
    # After a restart (new client, nothing bound) the saved state still works.
    fresh = _claude_agent(fake_cli)
    assert f"claude --resume {sid} --fork-session" in cc.render_handoff(fresh)
    agent.client.close()
    fresh.client.close()


def test_reset_drops_the_saved_session_even_without_a_bound_client(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli)
    _agent_turn(agent, fake_cli, HI)
    sid = agent.client._claude_session.conversation_ref()["session_id"]
    agent.client.close()
    fresh = _claude_agent(fake_cli)
    text = cc.do_reset(fresh)
    assert f"`{sid[:8]}`" in text
    assert not ccs._state_path("root|main").exists()
    assert "No Claude Code session to drop" in cc.do_reset(fresh)
    fresh.client.close()


def test_stop_interrupts_the_running_agent_and_the_claude_turn():
    from gateway import claude_code_commands as cc

    session = MagicMock()
    agent = SimpleNamespace(client=SimpleNamespace(_claude_session=session))
    running = MagicMock()
    text = cc.do_stop(agent, running)
    running.interrupt.assert_called_once_with()
    session.abort.assert_called_once_with()
    assert "Stopping Claude's reply" in text
    idle = MagicMock()
    idle.describe.return_value = {"busy": False}
    assert "not answering" in cc.do_stop(SimpleNamespace(client=SimpleNamespace(_claude_session=idle)), None)
    idle.abort.assert_not_called()


def test_doctor_reports_cli_login_and_bridge(fake_cli):
    from gateway import claude_code_commands as cc

    agent = _claude_agent(fake_cli)
    text = cc.render_doctor(agent)
    assert "Version: 9.9.9 (Claude Code)" in text
    assert "Login: ✅ logged in (claude.ai, firstParty, plan max)" in text
    assert "someone@example.com" not in text and "org-1" not in text
    assert "Fail-fast limit blocks: none" in text
    assert f"Working directory: `{fake_cli.tmp}`" in text
    agent.client.close()


def test_footer_meta_reports_cache_plan_and_turn_cost(fake_cli):
    from gateway import claude_code_commands as cc
    from gateway.runtime_footer import format_runtime_footer

    reset = int(time.time()) + 3600
    fake_cli.script({"rate_limit_info": _window_info(0.28, resets_at=reset, weekly=0.41), "cost": 0.42})
    agent = _claude_agent(fake_cli)
    before = cc.cost_total(agent)
    _agent_turn(agent, fake_cli, HI)
    meta = cc.footer_meta(agent, cost_before=before)
    assert meta["cache_pct"] == 90
    assert meta["turn_cost_usd"] == pytest.approx(0.42)
    assert meta["plan_5h"] == pytest.approx(0.28) and meta["plan_7d"] == pytest.approx(0.41)
    footer = format_runtime_footer(
        model=MODEL,
        context_tokens=10,
        context_length=100,
        fields=["model", "context_pct", "cache", "plan", "cost"],
        provider_meta=meta,
    )
    assert footer == "claude-opus-5 · 10% · cache 90% · 5h 28% · 7d 41% · ≈$0.42"
    assert cc.footer_meta(SimpleNamespace(provider="openrouter", client=None)) is None
    agent.client.close()


def test_footer_skips_provider_fields_without_meta():
    from gateway.runtime_footer import format_runtime_footer

    assert (
        format_runtime_footer(
            model="gpt-5.4", context_tokens=5, context_length=100,
            fields=["model", "cache", "plan", "cost"],
        )
        == "gpt-5.4"
    )


# ---------------------------------------------------------------------------
# Registration and gateway dispatch
# ---------------------------------------------------------------------------


def test_claude_command_is_registered_for_the_gateway():
    from hermes_cli.commands import (
        _SLACK_VIA_HERMES_ONLY,
        gateway_help_lines,
        resolve_command,
        should_bypass_active_session,
    )

    cmd = resolve_command("claude")
    assert cmd is not None and cmd.gateway_only and cmd.busy_policy == "dispatch"
    assert set(cmd.subcommands) == {
        "status", "usage", "context", "models", "stop", "reset", "doctor", "handoff"
    }
    assert should_bypass_active_session("claude")
    assert "claude" in _SLACK_VIA_HERMES_ONLY
    assert any(line.startswith("`/claude") for line in gateway_help_lines())
    assert not any(line.startswith("`/claude") for line in gateway_help_lines(hidden=("claude",)))


def _runner(agent=None, running=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = MagicMock()
    runner._session_db = None
    if running is not None:
        runner._running_agents["sk"] = running
    if agent is not None:
        runner._agent_cache["sk"] = (agent, "sig")
    runner._session_key_for_source = MagicMock(return_value="sk")
    runner._peek_session_state = MagicMock(return_value=None)
    return runner


def _event(args=""):
    event = MagicMock()
    event.get_command_args.return_value = args
    return event


def test_gateway_claude_command_is_gated_to_claude_code_sessions():
    runner = _runner(agent=SimpleNamespace(provider="openrouter", model="x", client=None))
    text = asyncio.run(runner._handle_claude_command(_event("status")))
    assert "`openrouter`" in text and "only while the session uses a Claude Code model" in text
    hidden = asyncio.run(runner._session_hidden_commands(_event()))
    assert hidden == ("claude",)


def test_gateway_claude_command_dispatches_subcommands(fake_cli):
    agent = _claude_agent(fake_cli)
    runner = _runner(agent=agent)
    assert asyncio.run(runner._session_hidden_commands(_event())) == ()
    assert "Claude Code models" in asyncio.run(runner._handle_claude_command(_event("models")))
    assert "Unknown `/claude nope`" in asyncio.run(runner._handle_claude_command(_event("nope")))
    assert "/claude handoff" in asyncio.run(runner._handle_claude_command(_event("help")))
    agent.client.close()


def test_gateway_claude_reset_waits_for_the_running_turn():
    agent = SimpleNamespace(provider="claude-code", model=MODEL, client=None)
    runner = _runner(running=agent)
    text = asyncio.run(runner._handle_claude_command(_event("reset")))
    assert "can't run mid-turn" in text
    busy = asyncio.run(runner._busy_claude_command(_event("reset"), "sk", None))
    assert "can't run mid-turn" in busy


def test_media_auto_append_names_results_by_their_own_tool():
    from gateway.run import _collect_auto_append_media_tags

    # A provider reused the id "c1" within one turn: the id map names only
    # the later terminal call, the TTS result still names itself.
    messages = [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "text_to_speech"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "text_to_speech",
         "tool_name": "text_to_speech", "content": "MEDIA:/tmp/hermes-tts/hi.ogg"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "terminal",
         "content": "printed MEDIA:/tmp/example.png"},
    ]
    tags, _voice = _collect_auto_append_media_tags(messages, history_offset=1)
    assert tags == ["MEDIA:/tmp/hermes-tts/hi.ogg"]


def test_gateway_help_lists_claude_only_in_claude_code_sessions(fake_cli):
    other = _runner(agent=SimpleNamespace(provider="openrouter", model="x", client=None))
    assert "/claude" not in asyncio.run(other._handle_help_command(_event()))
    assert "/claude" not in asyncio.run(other._handle_commands_command(_event("")))
    agent = _claude_agent(fake_cli)
    claude = _runner(agent=agent)
    assert "`/claude [status|" in asyncio.run(claude._handle_help_command(_event()))
    agent.client.close()


def test_gateway_provider_lookup_uses_the_persisted_model_override():
    runner = _runner()
    runner.session_store.get_model_override.return_value = {"provider": "claude-code", "model": MODEL}
    assert asyncio.run(runner._session_provider(_event())) == "claude-code"
    runner.session_store.get_or_create_session.assert_not_called()
