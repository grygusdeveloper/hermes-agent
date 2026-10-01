"""Claude subscription plan limits: the persisted snapshot, one-time warnings and hard limits.

Ported from the fork (73d5a85c74, a65491e5eb, 7025953856, 33d79181f5). Claude Code
emits ``rate_limit_event`` only when its ``rate_limit_info`` changes, so the latest
info is persisted next to the fork's durable session states in
``$HERMES_HOME/state/claude-code-sessions/rate_limit.json`` as
``{"info", "recorded_at", "warned"}`` (unchanged on-disk format, flock-protected),
where ``/status`` and ``/usage`` read it through :func:`read_rate_limit_snapshot`.

A plan window at 90% or more (or flagged ``allowed_warning`` by the CLI) is announced
once per window cycle; "announced" means a reply carried it to a user (see
``ClaudeCodeSession.take_notices``), so an auxiliary call that saw the window first
cannot swallow it.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.claude_code_errors import LIMIT_BANNER_RE, ClaudeCodeUsageLimitError

logger = logging.getLogger(__name__)

RATE_LIMIT_FILE_NAME = "rate_limit.json"
LIMIT_WARNING_UTILIZATION = 0.9
# After a hard limit, calls for the same model fail fast without starting the CLI
# until the reset, but a real call re-checks at least this often.
USAGE_LIMIT_RECHECK_SECONDS = 600.0
RATE_LIMIT_LABELS = {
    "five_hour": "5-hour session limit",
    "seven_day": "weekly limit",
    "seven_day_opus": "weekly Opus limit",
    "seven_day_sonnet": "weekly Sonnet limit",
    "seven_day_overage_included": "weekly limit",
    "overage": "extra-usage limit",
}

_FILE_LOCK = threading.Lock()
# Held across a read-modify-write of the file so this process's writers never
# drop each other's updates; the flock covers other processes.
_UPDATE_LOCK = threading.Lock()
_BLOCK_LOCK = threading.Lock()
# model -> (blocked until, rate_limit_info, CLI text)
_USAGE_LIMIT_BLOCKS: dict[str, tuple[float, dict[str, Any], str]] = {}


def state_dir() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "state" / "claude-code-sessions"


def rate_limit_path() -> Path:
    return state_dir() / RATE_LIMIT_FILE_NAME


@contextmanager
def _rate_limit_update():
    """Serialize a read-modify-write of ``rate_limit.json`` across threads and processes."""
    with _UPDATE_LOCK:
        directory = state_dir()
        fd = None
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(directory / "rate_limit.lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            if fd is not None:
                os.close(fd)
            fd = None
        try:
            yield
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def read_rate_limit_snapshot() -> dict[str, Any] | None:
    """The persisted plan snapshot ``{"info", "recorded_at", "warned"}``, or None. No side effects.

    ``info`` is the CLI's last ``rate_limit_info``, ``recorded_at`` when it arrived
    (epoch seconds) and ``warned`` the windows already announced (window -> reset epoch).
    """
    try:
        with _FILE_LOCK:
            payload = json.loads(rate_limit_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        return None
    recorded_at = payload.get("recorded_at")
    warned = payload.get("warned")
    return {
        "info": payload["info"],
        "recorded_at": float(recorded_at) if isinstance(recorded_at, (int, float)) else None,
        "warned": warned if isinstance(warned, dict) else {},
    }


def _save_snapshot(info: dict[str, Any], warned: dict[str, Any], *, recorded_at: float | None = None) -> None:
    path = rate_limit_path()
    encoded = json.dumps({"info": info, "recorded_at": time.time() if recorded_at is None else recorded_at,
                          "warned": warned}, ensure_ascii=True, separators=(",", ":"))
    with _FILE_LOCK:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def epoch_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value) / 1000.0 if value > 1e12 else float(value)


def _utilization(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def limit_window(info: dict[str, Any]) -> tuple[str | None, float | None]:
    """``(rateLimitType, reset epoch seconds)`` of a ``rate_limit_info``."""
    kind = info.get("rateLimitType") if isinstance(info.get("rateLimitType"), str) else None
    resets_at = epoch_seconds(info.get("resetsAt"))
    windows = info.get("unifiedWindows")
    if resets_at is None and kind and isinstance(windows, dict) and isinstance(windows.get(kind), dict):
        resets_at = epoch_seconds(windows[kind].get("resetsAt"))
    return kind, resets_at


def plan_windows(info: dict[str, Any] | None) -> dict[str, tuple[float | None, float | None]]:
    """``window -> (utilization fraction, reset epoch)`` of a ``rate_limit_info``."""
    windows: dict[str, tuple[float | None, float | None]] = {}
    if not isinstance(info, dict):
        return windows
    unified = info.get("unifiedWindows")
    if isinstance(unified, dict):
        for name, window in unified.items():
            if isinstance(name, str) and isinstance(window, dict):
                windows[name] = (_utilization(window.get("utilization")), epoch_seconds(window.get("resetsAt")))
    kind = info.get("rateLimitType")
    if isinstance(kind, str) and kind and kind not in windows:
        windows[kind] = (_utilization(info.get("utilization")), epoch_seconds(info.get("resetsAt")))
    return windows


def live_plan_windows(info: dict[str, Any] | None, now: float | None = None) -> dict[str, tuple[float | None, float | None]]:
    """:func:`plan_windows` minus the windows whose reset has passed."""
    now = time.time() if now is None else now
    return {name: window for name, window in plan_windows(info).items() if window[1] is None or window[1] > now}


def overage_usable(info: dict[str, Any]) -> bool:
    """Extra usage still serves requests once the plan window is exhausted."""
    return info.get("isUsingOverage") is True or str(info.get("overageStatus") or "") in {"allowed", "allowed_warning"}


def reset_label(resets_at: float | None, now: float | None = None) -> str:
    """``15:00 CEST (in ~2h 10m)`` in Hermes's timezone; "" when unknown."""
    if resets_at is None:
        return ""
    now = time.time() if now is None else now
    days, rest = divmod(max(0, int(resets_at - now)), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    span = f"{days}d {hours}h" if days else f"{hours}h {minutes:02d}m" if hours else f"{max(1, minutes)}m"
    try:
        try:
            from hermes_time import now as _hermes_now
            zone = _hermes_now().tzinfo
        except Exception:
            zone = None
        moment = datetime.fromtimestamp(resets_at, zone) if zone is not None else (
            datetime.fromtimestamp(resets_at).astimezone())
        today = datetime.fromtimestamp(now, moment.tzinfo).date()
        clock = moment.strftime("%H:%M" if moment.date() == today else "%a %H:%M")
        if moment.tzname():
            clock += f" {moment.tzname()}"
    except Exception:
        return f"in ~{span}"
    return f"{clock} (in ~{span})"


def _warned_for(warned: dict[str, Any], key: tuple[str, float | None]) -> bool:
    return key[0] in warned and warned[key[0]] == key[1]


def limit_warnings(info: dict[str, Any], warned: dict[str, Any], now: float) -> list[tuple[tuple[tuple[str, float | None], ...], str]]:
    """``(window keys, text)`` for plan limits to announce now.

    Windows naming the same limit (``seven_day`` and its twin
    ``seven_day_overage_included``) make one notice with both keys. A rejected
    request is reported by the error path instead.
    """
    if info.get("status") == "rejected":
        return []
    flagged = info.get("rateLimitType") if info.get("status") == "allowed_warning" else None
    by_label: dict[str, list[tuple[str, float | None, float | None]]] = {}
    for name, (utilization, resets_at) in plan_windows(info).items():
        hot = utilization is not None and utilization >= LIMIT_WARNING_UTILIZATION
        if (not hot and name != flagged) or (resets_at is not None and resets_at <= now):
            continue
        label = RATE_LIMIT_LABELS.get(name) or name.replace("_", " ") + " limit"
        by_label.setdefault(label, []).append((name, utilization, resets_at))
    found = []
    for label, windows in by_label.items():
        keys = tuple((name, resets_at) for name, _utilization, resets_at in windows)
        if any(_warned_for(warned, key) for key in keys):
            continue
        known = [item for item in windows if item[1] is not None]
        _name, utilization, resets_at = max(known, key=lambda item: item[1]) if known else windows[0]
        state = f"is {utilization:.0%} used" if utilization is not None else "is almost used up"
        when = reset_label(resets_at, now)
        found.append((keys, f"⚠️ Claude {label} {state}" + (f" — resets {when}" if when else "") + "."))
    return found


def record_rate_limit(info: Any) -> list[tuple[tuple[tuple[str, float | None], ...], str]]:
    """Persist a ``rate_limit_event``'s info; return the warnings it raises (not yet announced)."""
    if not isinstance(info, dict):
        return []
    try:
        with _rate_limit_update():
            snapshot = read_rate_limit_snapshot() or {}
            now = time.time()
            # Forget windows whose cycle is over.
            warned = {name: reset for name, reset in dict(snapshot.get("warned") or {}).items()
                      if not isinstance(reset, (int, float)) or reset > now}
            _save_snapshot(dict(info), warned)
    except Exception:
        logger.debug("Could not persist the Claude Code rate limit snapshot", exc_info=True)
        return []
    status = str(info.get("status") or "")
    hot = [f"{name}={utilization:.0%}" for name, (utilization, _reset) in plan_windows(info).items()
           if utilization is not None and utilization >= LIMIT_WARNING_UTILIZATION]
    if (status and status != "allowed") or hot:
        logger.warning("Claude Code rate limit: status=%s type=%s resets_at=%s %s", status or "?",
                       info.get("rateLimitType"), info.get("resetsAt"), " ".join(hot))
    return limit_warnings(info, warned, now)


def mark_limit_warnings_delivered(groups: list[tuple[tuple[str, float | None], ...]]) -> list[bool]:
    """Record plan warnings as shown, one group of window keys per notice.

    Returns, per group, whether it may be shown: one none of whose windows another
    session already announced this cycle. Those are recorded; the others dropped.
    """
    with _rate_limit_update():
        snapshot = read_rate_limit_snapshot()
        if snapshot is None:
            return [True for _group in groups]
        warned = dict(snapshot["warned"])
        verdicts = []
        for group in groups:
            fresh = not any(_warned_for(warned, key) for key in group)
            if fresh:
                warned.update(dict(group))
            verdicts.append(fresh)
        if any(verdicts):
            _save_snapshot(snapshot["info"], warned, recorded_at=snapshot.get("recorded_at"))
        return verdicts


# ── Hard subscription limits ────────────────────────────────────────────


def usage_limit_error(info: dict[str, Any], cli_text: str) -> ClaudeCodeUsageLimitError:
    """The error for a subscription limit that holds until its reset.

    ``info`` is the CLI's ``rate_limit_info`` for the rejection ({} when only the
    CLI's text says so); the window and reset come from it, else from the text.
    """
    kind, resets_at = limit_window(info)
    label = RATE_LIMIT_LABELS.get(kind or "", "")
    if not label:
        named = LIMIT_BANNER_RE.search(cli_text or "")
        label = f"{named.group(1)}limit" if named and named.group(1).strip() else "usage limit"
    detail = " ".join((cli_text or "").split())[:300]
    when = reset_label(resets_at)
    message = f"Claude {label} reached" + (f" — resets {when}" if when else "")
    if detail and not when:
        message += f" ({detail})"
    body = {"error": {
        "type": "usage_limit_reached", "message": message, "rate_limit_type": kind, "resets_at": resets_at,
        "resets_in_seconds": max(0, int(resets_at - time.time())) if resets_at else None,
    }}
    return ClaudeCodeUsageLimitError(message, body=body, detail=detail, resets_at=resets_at, rate_limit_type=kind)


def block_usage_limit(model: str, info: dict[str, Any], cli_text: str) -> None:
    now = time.time()
    until = now + USAGE_LIMIT_RECHECK_SECONDS
    _kind, resets_at = limit_window(info)
    if resets_at is not None:
        until = min(until, resets_at)
    if until <= now:
        return
    with _BLOCK_LOCK:
        _USAGE_LIMIT_BLOCKS[str(model)] = (until, dict(info), cli_text)


def usage_limit_block(model: str) -> ClaudeCodeUsageLimitError | None:
    """The error to fail with right away while ``model`` is at a subscription limit."""
    with _BLOCK_LOCK:
        entry = _USAGE_LIMIT_BLOCKS.get(str(model))
        if entry is None:
            return None
        if entry[0] <= time.time():
            _USAGE_LIMIT_BLOCKS.pop(str(model), None)
            return None
    return usage_limit_error(entry[1], entry[2])


def clear_usage_limit_blocks() -> None:
    with _BLOCK_LOCK:
        _USAGE_LIMIT_BLOCKS.clear()


# ── Showing notices with a reply ────────────────────────────────────────


def _shows_status_lines(agent: Any) -> bool:
    """Whether the agent's status lines reach a user: a gateway chat or the interactive CLI."""
    if callable(getattr(agent, "status_callback", None)):
        return True
    return str(getattr(agent, "platform", "") or "").strip().lower() == "cli"


def show_claude_code_notices(agent: Any, usage: Any) -> None:
    """Show the notices the Claude Code bridge attached to a response's usage.

    ``usage.hermes_claim_notices`` claims them (claiming records a plan warning as
    announced), so it is called only when this agent's status lines reach a user;
    otherwise they stay pending for a reply that is seen.
    """
    if (getattr(agent, "provider", "") or "").strip().lower() != "claude-code":
        return
    claim = getattr(usage, "hermes_claim_notices", None)
    if not callable(claim) or not _shows_status_lines(agent):
        return
    try:
        notices = claim()
    except Exception:
        logger.debug("Claude Code notices could not be claimed", exc_info=True)
        return
    for notice in notices if isinstance(notices, (list, tuple)) else ():
        if isinstance(notice, str) and notice.strip():
            agent._emit_status(notice.strip())
