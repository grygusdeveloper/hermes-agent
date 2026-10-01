"""Claude plan windows in /usage and the /status plan line (display part of 73d5a85c74 / a65491e5eb).

The plan snapshot is owned by agent.claude_code_limits; display reads it only through
``read_rate_limit_snapshot()``, replaced here by a stand-in module.
"""
import sys
import time
import types

import pytest

import agent.account_usage as account_usage


@pytest.fixture
def snapshot(monkeypatch):
    """Install ``agent.claude_code_limits.read_rate_limit_snapshot`` returning ``holder["value"]``."""
    holder = {"value": None, "calls": 0}

    def read_rate_limit_snapshot():
        holder["calls"] += 1
        return holder["value"]

    module = types.ModuleType("agent.claude_code_limits")
    module.read_rate_limit_snapshot = read_rate_limit_snapshot
    monkeypatch.setitem(sys.modules, "agent.claude_code_limits", module)
    return holder


def _window_info(utilization, *, resets_at, status="allowed", weekly=0.41):
    return {
        "status": status,
        "rateLimitType": "five_hour",
        "resetsAt": resets_at,
        "utilization": utilization,
        "unifiedWindows": {
            "five_hour": {"utilization": utilization, "resetsAt": resets_at},
            "seven_day": {"utilization": weekly, "resetsAt": resets_at + 86400},
        },
    }


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


def _text(snapshot):
    return "\n".join(account_usage.render_account_usage_lines(snapshot))


def test_rate_limit_snapshot_renders_fractions_status_and_age():
    info = _window_info(0.28, resets_at=time.time() + 3600, status="allowed_warning")
    info["overageStatus"] = "rejected"
    info["overageDisabledReason"] = "out_of_credits"
    text = _text(account_usage.build_claude_code_usage_snapshot(info, recorded_at=time.time() - 120))
    assert "Claude plan limits" in text and "Provider: claude-code" in text
    assert "Current session (5h): 72% remaining (28% used)" in text
    assert "Current week: 59% remaining (41% used)" in text
    assert text.index("Current session (5h)") < text.index("Current week")
    assert "Status: allowed warning (5-hour session limit)" in text
    assert "Extra usage: rejected (out of credits)" in text
    assert "As reported by Claude Code at" in text


def test_usage_marks_windows_that_reset_since_the_report():
    now = time.time()
    text = _text(account_usage.build_claude_code_usage_snapshot(_stale_info(now), recorded_at=now - 3 * 3600))
    assert "Current week: 59% remaining (41% used)" in text
    assert "Current session (5h): reset at" in text and "after this report" in text
    assert "0% remaining (100% used)" not in text
    assert "Status: rejected" not in text


def test_live_file_shape_with_seconds_and_milliseconds_resets():
    """The persisted info as the live host has it (``resetsAt`` in epoch seconds); older CLI builds
    reported milliseconds."""
    now = time.time()
    for scale in (1, 1000):
        info = {"status": "allowed_warning", "rateLimitType": "seven_day", "resetsAt": int((now + 86400) * scale),
                "utilization": 0.79, "isUsingOverage": False, "surpassedThreshold": 0.75,
                "unifiedWindows": {"five_hour": {"utilization": 0.25, "resetsAt": int((now + 3600) * scale)},
                                   "seven_day": {"utilization": 0.79, "resetsAt": int((now + 86400) * scale)}}}
        snapshot = account_usage.build_claude_code_usage_snapshot(info, now=now)
        assert {w.label: round(w.used_percent) for w in snapshot.windows} == {
            "Current session (5h)": 25, "Current week": 79}
        assert all(w.reset_at and w.reset_at.timestamp() > now for w in snapshot.windows)
        assert "Status: allowed warning (weekly limit)" in snapshot.details


def test_usage_for_claude_code_reads_the_persisted_snapshot(snapshot):
    snapshot["value"] = {"info": _window_info(0.5, resets_at=time.time() + 3600), "recorded_at": time.time(),
                         "warned": {}}
    result = account_usage.fetch_account_usage("claude-code", base_url="acp://claude-code", api_key="claude-code")
    assert result.provider == "claude-code"
    assert result.windows[0].used_percent == pytest.approx(50.0)


def test_usage_without_a_snapshot_or_limits_module_shows_nothing(snapshot, monkeypatch):
    assert account_usage.fetch_account_usage("claude-code") is None
    assert account_usage.claude_code_plan_line() == ""
    # The limits module failing (or absent) never breaks /usage or /status.
    monkeypatch.setitem(sys.modules, "agent.claude_code_limits", None)
    assert account_usage.fetch_account_usage("claude-code") is None
    assert account_usage.claude_code_plan_line() == ""


def test_plan_line_summarises_session_and_week(snapshot):
    snapshot["value"] = {"info": _window_info(0.28, resets_at=time.time() + 3600), "recorded_at": time.time(),
                         "warned": {}}
    line = account_usage.claude_code_plan_line()
    assert line.startswith("Claude plan: 5h 28% · week 41% (as of ")
    snapshot["value"]["info"]["status"] = "allowed_warning"
    assert "· near a limit" in account_usage.claude_code_plan_line()


def test_plan_line_skips_windows_that_reset_since_the_report(snapshot):
    snapshot["value"] = {"info": _stale_info(time.time()), "recorded_at": None, "warned": {}}
    line = account_usage.claude_code_plan_line()
    assert line == "Claude plan: week 41%"
    # Still within the rejected window: the limit shows.
    now = time.time()
    snapshot["value"]["info"] = _window_info(1.0, resets_at=now + 600, status="rejected")
    assert "5h 100%" in account_usage.claude_code_plan_line() and "limit reached" in account_usage.claude_code_plan_line()
