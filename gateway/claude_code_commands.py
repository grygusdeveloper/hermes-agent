"""``/claude``: Claude Code bridge views and controls for gateway sessions.

Only for sessions whose provider is ``claude-code`` (the local Claude Code
CLI, see ``agent/claude_code_session.py``):

* ``status``  — model, the Claude-side session, its warm process, what
  Claude is doing right now, the last call's usage and the plan windows.
* ``usage``   — Claude plan windows (5-hour session, weekly, per-model) with
  their resets, from Claude Code's ``get_usage``.
* ``context`` — Hermes's context meter next to Claude Code's own count.
* ``models``  — the models Claude Code offers (``list_models``).
* ``stop``    — stop the current reply gracefully (the session is kept).
* ``reset``   — drop only the Claude-side session; Hermes keeps the
  conversation and the next message replays it into a fresh one.
* ``doctor``  — CLI binary and version, login state, bridge state.
* ``handoff`` — the terminal command that forks this conversation into
  Claude Code.

The functions are synchronous: the gateway calls them off the event loop.
Control requests go to the session's parked warm process, else to a throwaway
control-only CLI process; none of them starts a model turn. Nothing here
reads or prints OAuth tokens.
"""

from __future__ import annotations

import logging
import os
import shlex
import time
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

CLAUDE_CODE_PROVIDER = "claude-code"
SUBCOMMANDS: tuple[str, ...] = (
    "status",
    "usage",
    "context",
    "models",
    "stop",
    "reset",
    "doctor",
    "handoff",
)
# ``reset`` would race the running turn's own session state.
BUSY_SAFE_SUBCOMMANDS = frozenset(SUBCOMMANDS) - {"reset"}
BUSY_RESET_TEXT = (
    "⏳ Claude is answering — `/claude reset` can't run mid-turn. "
    "Wait for the reply or `/claude stop` first."
)

# How long ``/claude reset`` waits for a request still holding the session
# (an abandoned reply winding down) before it asks the user to retry.
_RESET_WAIT_SECONDS = 20.0

# Plan windows in the one-line summaries, in display order.
_PLAN_SHORT_LABELS: tuple[tuple[str, str], ...] = (
    ("five_hour", "5h"),
    ("seven_day", "week"),
    ("seven_day_opus", "Opus week"),
    ("seven_day_sonnet", "Sonnet week"),
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def parse_subcommand(args: Optional[str]) -> str:
    """The subcommand of ``/claude <args>`` (``status`` when empty)."""

    parts = (args or "").strip().split(None, 1)
    return parts[0].lower() if parts else "status"


def session_provider(agent: Any) -> str:
    """The provider this session runs on.

    While a turn is served by a fallback provider, ``agent.provider`` names
    the fallback; the session still belongs to its primary provider.
    """

    if agent is None:
        return ""
    if getattr(agent, "_fallback_activated", False):
        primary = getattr(agent, "_primary_runtime", None)
        if isinstance(primary, dict) and primary.get("provider"):
            return str(primary["provider"]).strip().lower()
    return str(getattr(agent, "provider", "") or "").strip().lower()


def claude_client(agent: Any) -> Any:
    """The agent's Claude Code client (request clients share its session)."""

    client = getattr(agent, "client", None) if agent is not None else None
    return client if getattr(client, "_claude_session", None) is not None else None


def claude_session(agent: Any) -> Any:
    client = claude_client(agent)
    return client._claude_session if client is not None else None


def state_keys(agent: Any) -> list[str]:
    """Durable bridge state keys of this agent's conversation.

    The key the session last used first, then the one the agent's next
    request will use (they differ after a compression rotated the session).
    """

    keys: list[str] = []
    session = claude_session(agent)
    bound = getattr(session, "_state_key", None) if session is not None else None
    if isinstance(bound, str) and bound:
        keys.append(bound)
    if agent is not None:
        try:
            from run_agent import _bridge_state_key_resolver

            root_of = getattr(agent, "_conversation_root_id", None)
            root = root_of() if callable(root_of) else None
            resolver = _bridge_state_key_resolver(agent, root)
            key = resolver() if resolver else None
        except Exception:
            logger.debug("Could not resolve the Claude Code state key", exc_info=True)
            key = None
        if isinstance(key, str) and key and key not in keys:
            keys.append(key)
    return keys


def not_claude_text(provider: str) -> str:
    where = f"`{provider}`" if provider else "another provider"
    return (
        f"This session runs on {where}. `/claude` works only while the session "
        "uses a Claude Code model (switch with `/model`)."
    )


def help_text() -> str:
    return "\n".join(
        [
            "**/claude** — Claude Code bridge",
            "`/claude status` — model, Claude session, warm process, last call, plan",
            "`/claude usage` — plan limits (5-hour session, weekly) and resets",
            "`/claude context` — context size as Hermes and Claude Code count it",
            "`/claude models` — models Claude Code offers",
            "`/claude stop` — stop the current reply (the session is kept)",
            "`/claude reset` — drop only the Claude-side session (history stays)",
            "`/claude doctor` — CLI version, login and bridge health",
            "`/claude handoff` — continue this conversation in a terminal",
        ]
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _tokens(value: Any) -> str:
    number = _number(value) or 0.0
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if number >= 1000:
        return f"{number / 1000:.1f}k"
    return str(int(number))


def _clock(epoch: float, now: Optional[float] = None) -> str:
    """``14:02`` today, ``Mon 14:02`` otherwise, in Hermes's timezone."""

    try:
        try:
            from hermes_time import now as _hermes_now

            zone = _hermes_now().tzinfo
        except Exception:
            zone = None
        moment = datetime.fromtimestamp(epoch, zone) if zone else datetime.fromtimestamp(epoch).astimezone()
        today = datetime.fromtimestamp(time.time() if now is None else now, moment.tzinfo).date()
        return moment.strftime("%H:%M" if moment.date() == today else "%a %H:%M")
    except Exception:
        return "?"


def plan_snapshot(session: Any = None) -> tuple[dict[str, Any], Optional[float]]:
    """``(rate_limit_info, recorded_at)``: the persisted snapshot (at least as
    recent as any session's copy), else ``session``'s in-memory one."""

    from agent.claude_code_session import load_rate_limit_snapshot

    persisted = load_rate_limit_snapshot()
    if persisted is not None:
        return dict(persisted["info"]), persisted.get("recorded_at")
    info = getattr(session, "last_rate_limit", None) if session is not None else None
    return (dict(info) if isinstance(info, dict) else {}), None


def plan_line(session: Any = None) -> str:
    """``Claude plan: 5h 28% · week 41% (as of 14:02)``, or "" without data."""

    from agent.claude_code_session import live_plan_windows, plan_status_live

    info, recorded_at = plan_snapshot(session)
    if not info:
        return ""
    # Windows that reset since the report no longer have its figures.
    windows = live_plan_windows(info)
    parts = []
    for key, label in _PLAN_SHORT_LABELS:
        utilization = windows.get(key, (None, None))[0]
        if utilization is not None:
            parts.append(f"{label} {utilization:.0%}")
    status = str(info.get("status") or "") if plan_status_live(info) else ""
    if status == "rejected":
        parts.append("limit reached")
    elif status == "allowed_warning":
        parts.append("near a limit")
    if not parts:
        return ""
    text = "Claude plan: " + " · ".join(parts)
    if recorded_at:
        text += f" (as of {_clock(recorded_at)})"
    return text


def _last_call_line(usage: dict[str, Any]) -> str:
    prompt = _number(usage.get("prompt_tokens")) or 0.0
    cached = _number(usage.get("cached_tokens")) or 0.0
    parts = [f"{_tokens(prompt)} in" + (f" ({round(100 * cached / prompt)}% cached)" if prompt else "")]
    parts.append(f"{_tokens(usage.get('output_tokens'))} out")
    api_ms = _number(usage.get("duration_api_ms"))
    if api_ms is not None:
        parts.append(f"API {api_ms / 1000:.1f}s")
    cost = _number(usage.get("total_cost_usd"))
    if cost is not None:
        parts.append(f"≈${cost:.2f} API-equivalent")
    served = usage.get("served_model")
    if isinstance(served, str) and served:
        parts.append(f"answered by {served}")
    return "Last call: " + " · ".join(parts)


def _effort_label(agent: Any, bound: Optional[str]) -> str:
    config = getattr(agent, "reasoning_config", None)
    if isinstance(config, dict) and config.get("enabled") is False:
        return "thinking off"
    effort = bound or (config.get("effort") if isinstance(config, dict) else None)
    return f"effort {effort}" if isinstance(effort, str) and effort else ""


def _saved_ref(agent: Any) -> Optional[dict[str, Any]]:
    """The conversation to hand off / inspect: the bound one, else a saved one."""

    session = claude_session(agent)
    keys = state_keys(agent)
    if session is not None:
        ref = session.conversation_ref(keys[0] if keys else None)
        if ref is not None:
            return ref
    from agent.claude_code_session import _load_durable_state

    for key in keys:
        durable = _load_durable_state(key)
        if durable is not None:
            return {"session_id": durable.session_id, "model": durable.model, "persisted": True}
    return None


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def render_status(agent: Any) -> str:
    session = claude_session(agent)
    lines = ["🟠 **Claude Code**"]
    info = session.describe() if session is not None else {}
    model = getattr(agent, "model", "") if agent is not None else ""
    effort = _effort_label(agent, info.get("effort"))
    lines.append(f"Model: `{model or '?'}`" + (f" · {effort}" if effort else ""))
    if session is None:
        lines.append(
            "Bridge: not loaded right now (no agent yet, or a fallback provider is serving)"
        )
    else:
        sid = info.get("session_id")
        if sid:
            kind = "saved" if info.get("persisted") else "in memory only"
            lines.append(f"Claude session: `{sid[:8]}` ({kind} · has seen {info.get('messages', 0)} messages)")
        else:
            ref = _saved_ref(agent)
            if ref is not None:
                lines.append(f"Claude session: `{ref['session_id'][:8]}` (saved; loaded on the next message)")
            else:
                lines.append("Claude session: none yet (the next message starts one)")
        warm = info.get("warm")
        if info.get("busy"):
            process = "answering now"
        elif warm and warm.get("parked"):
            process = f"warm and idle (pid {warm.get('pid')}, {warm.get('turns', 0)} turns)"
        else:
            process = "none (the next reply resumes the saved session)"
        lines.append(f"Process: {process}")
        progress = info.get("progress") or {}
        if progress.get("active") and progress.get("description"):
            lines.append(f"Now: {progress['description']}")
        if info.get("interrupted"):
            lines.append("The last request was interrupted; the next message continues after it")
        usage = info.get("last_usage") or {}
        if usage:
            lines.append(_last_call_line(usage))
    plan = plan_line(session)
    if plan:
        lines.append(plan)
    lines.extend(["", "More: `/claude usage` · `context` · `models` · `doctor` · `handoff` · `stop` · `reset`"])
    return "\n".join(lines)


def render_usage(agent: Any) -> str:
    from agent.account_usage import fetch_claude_code_account_usage, render_account_usage_lines

    snapshot = fetch_claude_code_account_usage(claude_client(agent))
    lines = render_account_usage_lines(snapshot, markdown=True)
    if not lines:
        return (
            "No Claude plan data yet: Claude Code did not answer `get_usage`, and it "
            "has not reported its limits with a reply yet."
        )
    return "\n".join(lines)


def render_context(agent: Any) -> str:
    lines = ["🧠 **Context — Hermes and Claude Code**"]
    compressor = getattr(agent, "context_compressor", None) if agent is not None else None
    used = int(_number(getattr(compressor, "last_prompt_tokens", 0)) or 0)
    window = int(_number(getattr(compressor, "context_length", 0)) or 0)
    threshold = int(_number(getattr(compressor, "threshold_tokens", 0)) or 0)
    if window:
        pct = round(100 * used / window)
        text = f"Hermes: {used:,} / {window:,} tokens ({pct}%)"
        if threshold:
            text += f" · compresses at {threshold:,}"
        lines.append(text)
    else:
        lines.append("Hermes: no measurement yet")
    client = claude_client(agent)
    if client is None:
        lines.append("Claude Code: the bridge is not loaded right now")
        return "\n".join(lines)
    keys = state_keys(agent)
    try:
        answer = client.control_requests(
            [("get_context_usage", {"detail": "summary"})],
            conversation=True,
            state_key=keys[0] if keys else None,
        )[0]
    except Exception as exc:
        answer = exc
    if isinstance(answer, BaseException):
        lines.append(f"Claude Code: unavailable — {answer}")
        return "\n".join(lines)
    total = _number(answer.get("totalTokens"))
    maximum = _number(answer.get("maxTokens"))
    if total is not None and maximum:
        text = f"Claude Code: {int(total):,} / {int(maximum):,} tokens ({round(100 * total / maximum)}%)"
        if answer.get("model"):
            text += f" · `{answer['model']}`"
        lines.append(text)
    categories = [
        (str(item.get("name") or "?"), int(_number(item.get("tokens")) or 0))
        for item in answer.get("categories") or []
        # "free" space and the "buffer" Claude Code reserves are not content.
        if isinstance(item, dict)
        and item.get("kind", "used") == "used"
        and (_number(item.get("tokens")) or 0) > 0
    ]
    for name, tokens in categories[:8]:
        lines.append(f"• {name}: {tokens:,}")
    lines.append(
        "Claude Code counts what the CLI holds (Hermes's instructions and tool "
        "schemas included); Hermes compresses by its own count."
    )
    return "\n".join(lines)


def _control(agent: Any, requests: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Control requests through the agent's client, else a throwaway CLI."""

    client = claude_client(agent)
    try:
        if client is not None:
            return client.control_requests(requests)
        from agent.claude_code_client import _build_subprocess_env, _resolve_command
        from agent.claude_code_session import run_control_requests

        return run_control_requests(requests, command=_resolve_command(), env=_build_subprocess_env())
    except Exception as exc:
        return [exc for _request in requests]


def render_models(agent: Any) -> str:
    answer = _control(agent, [("list_models", {})])[0]
    if isinstance(answer, BaseException):
        return f"Claude Code could not list its models: {answer}"
    models = [item for item in answer.get("models") or [] if isinstance(item, dict)]
    if not models:
        return "Claude Code listed no models."
    current = str(getattr(agent, "model", "") or "") if agent is not None else ""
    lines = ["🧩 **Claude Code models**"]
    for item in models:
        value = str(item.get("value") or "?")
        resolved = str(item.get("resolvedModel") or "")
        name = str(item.get("displayName") or value)
        line = f"• `{value}` — {name}"
        if resolved and resolved != value:
            line += f" (`{resolved}`)"
        description = str(item.get("description") or "").strip()
        if description:
            line += f": {description}"
        efforts = item.get("supportedEffortLevels")
        if isinstance(efforts, list) and efforts:
            line += f" · effort {'/'.join(str(level) for level in efforts)}"
        if current and current in {value, resolved}:
            line += " ← this session"
        lines.append(line)
    return "\n".join(lines)


def render_doctor(agent: Any) -> str:
    from agent.claude_code_client import _build_subprocess_env, _resolve_command
    from agent.claude_code_session import bridge_diagnostics, claude_cli_diagnostics

    client = claude_client(agent)
    command = getattr(client, "_claude_command", None) or _resolve_command()
    facts = claude_cli_diagnostics(command, env=_build_subprocess_env())
    bridge = bridge_diagnostics()
    lines = ["🩺 **Claude Code doctor**"]
    binary = f"CLI: `{facts['command']}`"
    if facts.get("path") and facts["path"] != facts["command"]:
        binary += f" → `{facts['path']}`"
    lines.append(binary)
    lines.append(f"Version: {facts.get('version') or 'unknown'}")
    auth = facts.get("auth") or {}
    if auth:
        state = "✅ logged in" if auth.get("loggedIn") else "❌ not logged in"
        extra = [
            str(auth[name])
            for name in ("authMethod", "apiProvider")
            if isinstance(auth.get(name), str) and auth[name]
        ]
        if isinstance(auth.get("subscriptionType"), str) and auth["subscriptionType"]:
            extra.append(f"plan {auth['subscriptionType']}")
        lines.append("Login: " + state + (f" ({', '.join(extra)})" if extra else ""))
    else:
        lines.append("Login: unknown")
    for error in facts.get("errors") or []:
        lines.append(f"⚠️ {error}")
    if client is not None:
        lines.append(f"Working directory: `{client._claude_cwd}`")
    flags = [
        f"{name} {'on' if ok else 'off (the CLI rejected it)'}"
        for name, ok in (bridge.get("flags") or {}).items()
    ]
    if flags:
        lines.append("Optional flags: " + ", ".join(flags))
    lines.append(
        f"Warm processes: {bridge['warm_parked']}/{bridge['warm_cap']} parked · "
        f"idle lifetime {bridge['keepalive_seconds']:.0f}s"
    )
    blocks = bridge.get("usage_limit_blocks") or {}
    if blocks:
        lines.append(
            "Fail-fast limit blocks: "
            + ", ".join(f"`{model}` until {_clock(until)}" for model, until in blocks.items())
        )
    else:
        lines.append("Fail-fast limit blocks: none")
    lines.append(f"Saved Claude sessions: {bridge['state_files']} in `{bridge['state_dir']}`")
    plan = plan_line(claude_session(agent))
    if plan:
        lines.append(plan)
    return "\n".join(lines)


def render_handoff(agent: Any) -> str:
    ref = _saved_ref(agent)
    if ref is None:
        return "No Claude Code conversation is saved for this session yet — send a message first."
    if not ref.get("persisted"):
        return (
            "This Claude Code conversation exists only inside its running process "
            "(not saved on disk), so it cannot be handed off."
        )
    client = claude_client(agent)
    cwd = getattr(client, "_claude_cwd", None) or os.getcwd()
    command = f"cd {shlex.quote(cwd)} && claude --resume {ref['session_id']} --fork-session"
    return "\n".join(
        [
            "🤝 **Continue this conversation in a terminal**",
            "```",
            command,
            "```",
            "`--fork-session` gives the terminal its own copy, so this chat keeps the "
            "original. The copy holds the conversation as Claude saw it, up to its "
            "last reply; Claude Code finds it only from that directory.",
        ]
    )


def do_reset(agent: Any) -> str:
    from agent.claude_code_session import ClaudeCodeSession

    session = claude_session(agent)
    target = session if session is not None else ClaudeCodeSession()
    keys = state_keys(agent)
    dropped: list[str] = []
    for key in keys or [None]:
        try:
            sid = target.reset_conversation(key, wait=_RESET_WAIT_SECONDS)
        except TimeoutError:
            return (
                "⏳ Claude Code is still finishing a request for this session — "
                "try `/claude reset` again in a moment (or `/claude stop` first)."
            )
        if sid and sid not in dropped:
            dropped.append(sid)
    if not dropped:
        return "No Claude Code session to drop; the next message starts a fresh one anyway."
    ids = ", ".join(f"`{sid[:8]}`" for sid in dropped)
    return (
        f"🧹 Dropped the Claude Code session {ids}. Hermes keeps this conversation; "
        "the next message starts a fresh Claude session from it (one full replay)."
    )


def do_stop(agent: Any, running_agent: Any) -> str:
    session = claude_session(agent)
    if running_agent is not None:
        try:
            # The agent's abort hook interrupts the Claude Code turn
            # gracefully (its warm process stays parked).
            running_agent.interrupt()
        except Exception:
            logger.debug("Agent interrupt for /claude stop failed", exc_info=True)
        if session is not None:
            session.abort()
        return (
            "⏹ Stopping Claude's reply. The Claude session is kept; your next "
            "message continues after the interrupted reply."
        )
    if session is not None and session.describe().get("busy"):
        session.abort()
        return "⏹ Stopped the running Claude Code request."
    return "Claude is not answering anything right now."


RENDERERS = {
    "status": render_status,
    "usage": render_usage,
    "context": render_context,
    "models": render_models,
    "doctor": render_doctor,
    "handoff": render_handoff,
    "reset": do_reset,
}


# ---------------------------------------------------------------------------
# Runtime footer
# ---------------------------------------------------------------------------


def cost_total(agent: Any) -> Optional[float]:
    """The session's running API-equivalent cost (diffed around a turn)."""

    session = claude_session(agent)
    value = getattr(session, "cost_total_usd", None) if session is not None else None
    return float(value) if isinstance(value, (int, float)) else None


def footer_meta(agent: Any, *, cost_before: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Claude fields for the runtime footer (``gateway/runtime_footer.py``).

    ``cache_pct`` (cache reads of the last call's prompt), ``turn_cost_usd``
    (API-equivalent cost of the turn's CLI calls), ``plan_5h``/``plan_7d``
    (window utilization as a fraction) and ``served_model`` after a model
    fallback. None unless the session runs on Claude Code.
    """

    if session_provider(agent) != CLAUDE_CODE_PROVIDER:
        return None
    session = claude_session(agent)
    if session is None:
        return None
    from agent.claude_code_session import live_plan_windows

    meta: dict[str, Any] = {}
    usage = session.last_usage
    prompt = _number(usage.get("prompt_tokens")) or 0.0
    if prompt > 0:
        meta["cache_pct"] = round(100 * (_number(usage.get("cached_tokens")) or 0.0) / prompt)
    after = cost_total(agent)
    # Nothing added: no CLI turn ran, or its share of a restored session's
    # total was unknown. Leave the cost out rather than show $0.00.
    if cost_before is not None and after is not None and after > cost_before:
        meta["turn_cost_usd"] = after - cost_before
    info, _recorded_at = plan_snapshot(session)
    windows = live_plan_windows(info)
    for key, field in (("five_hour", "plan_5h"), ("seven_day", "plan_7d")):
        utilization = windows.get(key, (None, None))[0]
        if utilization is not None:
            meta[field] = utilization
    served = usage.get("served_model")
    if isinstance(served, str) and served:
        meta["served_model"] = served
    return meta or None
