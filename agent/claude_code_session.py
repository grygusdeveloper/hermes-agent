"""Durable Claude Code CLI conversation transport.

The generic ACP compatibility client (``CopilotACPClient``) flattens the full
OpenAI-style message history into one prompt and relaunches a short-lived
subprocess on every turn.  ``claude`` (Claude Code CLI 2.x) exposes a richer
native transport: ``--input-format stream-json`` accepts a single JSON-encoded
user message on stdin and ``--output-format stream-json`` emits typed events
(``system``, ``assistant``, ``user`` tool-result, ``result``).

This module keeps one durable Claude Code **session** per cached Hermes model
client.  After the first request establishes a ``session_id`` (a stable UUID),
subsequent turns resume that same server-side conversation with
``--resume <session_id>`` and send only the new non-assistant messages.  This
preserves the complete canonical Hermes request (system prompt, tool schemas,
full transcript) on the first turn while keeping later turns incremental and
bounded — analogous to ``agent/antigravity_session.py`` but using Claude Code's
native stdin transport instead of an argv-bound prompt.

Design invariants
-----------------
* **Native stream-json stdin** — the prompt/content travels on stdin, never as
  an argv ``execve`` argument, so there is no ``MAX_ARG_STRLEN`` ceiling.
* **Tools disabled** — Claude Code's own built-in tools are turned off
  (``--tools ""``).  All tool execution stays under Hermes logging,
  permissions, MCP, and approvals.  Hermes injects its tool schemas into the
  prompt and parses ``<tool_call>`` blocks back out of the response.
* **Exact model + effort** — ``--model`` and ``--effort`` are forwarded on
  every turn.
* **No credential exposure** — authentication is delegated entirely to the
  already-authenticated Claude Code CLI; this module never reads, passes, or
  logs credentials.
* **Cancellation / timeout / cleanup** — every request owns its process via a
  process-group latch; aborts and timeouts kill the whole group and reap it.
* **Expired-session recovery** — a missing/expired server session is detected
  and retried once as a fresh conversation with the complete prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# Claude Code's stream-json output requires ``--verbose``; we always pass it.
# Linux limits each execve argument to MAX_ARG_STRLEN (normally 128 KiB).
# Because we transport the prompt over stdin (not argv), this limit applies
# only to the short CLI flags and is never a practical constraint.  We keep a
# guard anyway so an accidental argv prompt is rejected loudly.
_INLINE_FLAG_LIMIT_BYTES = 120_000
# Claude Code session IDs are UUIDs.
_SESSION_ID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_MESSAGE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_VERSION = 2
_STATE_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)

# Markers that indicate the Claude Code server session is gone / expired.
_EXPIRED_SESSION_MARKERS = (
    "session not found",
    "invalid session",
    "session expired",
    "session has expired",
    "unknown session",
    "no such session",
    "could not resume",
    "resume failed",
    "session does not exist",
    # Exact Claude Code CLI 2.1.x wording observed on invalid --resume:
    #   "No conversation found with session ID: <uuid>"
    "no conversation found with session id",
    "no conversation found",
    "conversation not found",
)


def _resolve_claude_command() -> str:
    """Resolve the Claude Code CLI binary path.

    Never invents a path: honours an explicit override then falls back to PATH
    lookup of ``claude``.  Raises if the binary cannot be found so callers
    surface a clear error instead of a confusing FileNotFoundError.
    """

    explicit = (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_PATH", "").strip()
    )
    if explicit:
        return explicit
    found = os.getenv("CLAUDE_BIN", "").strip()
    if found:
        return found
    return "claude"


def _render_content(content: Any) -> str:
    """Render an OpenAI-style message ``content`` field to plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text not in (None, ""):
                    parts.append(str(text))
            elif item not in (None, ""):
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def _normalize_for_digest(value: Any) -> Any:
    """Convert structured message fields to stable JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_digest(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_digest(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _normalize_for_digest(model_dump())
        except Exception:
            pass
    return str(value)


def _message_fingerprint(messages: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Return exact structural prefix identity without retaining prompt text."""

    fingerprints: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        durable_content = _render_content(message.get("content"))
        trusted_oob = message.get("_hermes_oob_user_message")
        if (
            isinstance(trusted_oob, str)
            and trusted_oob
            and durable_content.endswith(trusted_oob)
        ):
            durable_content = durable_content[: -len(trusted_oob)]
        identity = {
            "role": role,
            "content": durable_content,
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": _normalize_for_digest(message.get("tool_calls")),
        }
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        fingerprints.append(
            (role, hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        )
    return tuple(fingerprints)


def _state_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home

        hermes_home = Path(get_hermes_home())
    except Exception:
        hermes_home = Path(
            os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
        )
    return hermes_home / "state" / "claude-code-sessions"


def _state_path(state_key: str) -> Path:
    digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
    return _state_dir() / f"{digest}.json"


@contextmanager
def _durable_transition_lock(state_key: str | None):
    """Serialize one session's load -> dispatch -> publication transition."""

    if not state_key:
        yield
        return
    directory = _state_dir()
    digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
    lock_path = directory / f"{digest}.lock"
    with _STATE_LOCK:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_durable_state(
    state_key: str,
) -> tuple[str, tuple[tuple[str, str], ...], str, str | None, str] | None:
    path = _state_path(state_key)
    try:
        with _STATE_LOCK:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _STATE_VERSION:
        return None
    expected_key_hash = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
    if payload.get("state_key_hash") != expected_key_hash:
        return None
    session_id = payload.get("session_id")
    raw_fingerprints = payload.get("message_fingerprints")
    model = payload.get("model")
    effort = payload.get("effort")
    tools_digest = payload.get("tools_digest")
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    if effort is not None and not isinstance(effort, str):
        return None
    if not isinstance(tools_digest, str):
        return None
    if not isinstance(raw_fingerprints, list):
        return None
    fingerprints: list[tuple[str, str]] = []
    for item in raw_fingerprints:
        if not isinstance(item, list) or len(item) != 2:
            return None
        role, digest = item
        if not isinstance(role, str) or not isinstance(digest, str):
            return None
        if not _MESSAGE_DIGEST_RE.fullmatch(digest):
            return None
        fingerprints.append((role, digest))
    return (
        session_id,
        tuple(fingerprints),
        model.strip(),
        effort.strip() if isinstance(effort, str) and effort.strip() else None,
        tools_digest,
    )


def _save_durable_state(
    state_key: str,
    session_id: str,
    fingerprints: tuple[tuple[str, str], ...],
    *,
    model: str,
    effort: str | None,
    tools_digest: str,
) -> None:
    directory = _state_dir()
    path = _state_path(state_key)
    payload = {
        "version": _STATE_VERSION,
        "state_key_hash": hashlib.sha256(state_key.encode("utf-8")).hexdigest(),
        "session_id": session_id,
        "message_fingerprints": [list(item) for item in fingerprints],
        "model": model,
        "effort": effort,
        "tools_digest": tools_digest,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    with _STATE_LOCK:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        temp = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def _delete_durable_state(state_key: str) -> None:
    with _STATE_LOCK:
        try:
            _state_path(state_key).unlink()
        except FileNotFoundError:
            pass


def _validate_flag_size(text: str, limit: int = _INLINE_FLAG_LIMIT_BYTES) -> str:
    """Reject an oversized exec argument instead of silently losing context.

    The prompt is transported over stdin so this guard only ever fires on a
    programming error (e.g. accidentally passing the prompt as a CLI flag).
    """

    size = len(text.encode("utf-8"))
    if size > limit:
        raise RuntimeError(
            "Claude Code inline flag limit exceeded: "
            f"{size} UTF-8 bytes > {limit}; prompt must travel over stdin"
        )
    return text


def _incremental_prompt(messages: list[dict[str, Any]], previous_count: int) -> str:
    """Render only the new non-assistant messages for a resumed session."""

    parts: list[str] = []
    for message in messages[previous_count:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        # Claude Code already owns its prior assistant output in the server
        # session.  Re-sending it would duplicate context.
        if role == "assistant":
            continue
        rendered = _render_content(message.get("content"))
        if not rendered and role != "tool":
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("id") or ""
            tool_name = message.get("name") or ""
            header = "Tool Result"
            meta_bits = []
            if isinstance(tool_name, str) and tool_name.strip():
                meta_bits.append(f"name={tool_name.strip()}")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                meta_bits.append(f"tool_call_id={tool_call_id.strip()}")
            if meta_bits:
                header = f"Tool Result ({', '.join(meta_bits)})"
            parts.append(f"{header}:\n{rendered}")
            continue
        label = {
            "system": "System",
            "user": "User",
        }.get(role, role.title() or "Context")
        parts.append(f"{label}:\n{rendered}")
    return "\n\n".join(parts)


class ClaudeCodeSessionExpired(RuntimeError):
    """Claude Code positively identified a missing/invalid/expired session."""


class ClaudeCodeSoftLimitNotice(RuntimeError):
    """Claude Code returned a soft billing/limit notice as successful content.

    This is a *warning*, not a hard transport failure. Callers should retry the
    same turn rather than treating the notice text as the model answer.
    """


# Soft notices Claude Code may emit as a normal successful ``result`` string.
# Observed live: "You've hit your monthly spend limit · raise it at
# claude.ai/settings/usage" while rate_limit_info still reports allowed.
_SOFT_LIMIT_MARKERS = (
    "you've hit your monthly spend limit",
    "hit your monthly spend limit",
    "monthly spend limit",
    "raise it at claude.ai/settings/usage",
    "claude.ai/settings/usage",
    "out of extra usage",
    "you're out of extra usage",
    "usage limit reached",
    "hit your usage limit",
    "rate limit reached",
    "too many requests",
)


def _is_soft_limit_detail(detail: str) -> bool:
    """Return True when a Claude Code error detail indicates a soft/rate limit.

    Claude Code returns rate-limit / spend-limit conditions as exit 1 with
    ``is_error:true`` and either an explicit ``rate_limit`` error type or an
    ``api_error_status`` of 429, often accompanied by the "monthly spend limit"
    banner.  These are transient and should be retried, not surfaced as a hard
    failure.
    """

    if not detail:
        return False
    lower = detail.lower()
    # Direct protocol signals from the stream-json result event.
    if '"error":"rate_limit"' in lower or '"error": "rate_limit"' in lower:
        return True
    if 'api_error_status":429' in lower or 'api_error_status": 429' in lower:
        return True
    # Known soft-limit banner text embedded in the result.
    return any(marker in lower for marker in _SOFT_LIMIT_MARKERS)


def _is_expired_session_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(marker in normalized for marker in _EXPIRED_SESSION_MARKERS)


def _is_soft_limit_notice(text: str) -> bool:
    """Return True when CLI result text is a soft limit/billing notice.

    Only match short, notice-like replies so normal answers that merely
    *mention* usage settings are not treated as failures.
    """

    body = (text or "").strip()
    if not body:
        return False
    # Real model answers are rarely pure one-line billing banners.
    if len(body) > 600:
        return False
    normalized = body.lower()
    if not any(marker in normalized for marker in _SOFT_LIMIT_MARKERS):
        return False
    # Prefer high-confidence patterns: short banner-like lines.
    line_count = body.count("\n") + 1
    if line_count <= 4:
        return True
    # Multi-line but almost entirely the notice (no substantial extra prose).
    non_empty = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return len(non_empty) <= 6


# Short "I'm about to start" preambles that the CLI sometimes returns as a
# complete successful result when the turn was cut short mid-process.
_INCOMPLETE_PREAMBLE_STARTERS = (
    "i'll ",
    "i will ",
    "let me ",
    "i'm going to ",
    "i am going to ",
    "i'm about to ",
    "checking ",
    "looking ",
    "analyzing ",
    "investigating ",
    "implementing ",
    "applying ",
    "fixing ",
    "working on ",
    "processing ",
    "doing a ",
    "i'll do ",
    "i will do ",
)

_PROGRESS_CONTINUATION_PROMPT = """\
Continue the previous request now. Your last reply was only a progress/status
statement, not a completed answer. Do not repeat the plan. If information or
an action is needed, immediately emit the appropriate Hermes <tool_call> block.
Otherwise return the complete user-facing result now.
""".strip()


def _is_incomplete_preamble_response(
    text: str,
    *,
    had_tools: bool,
    has_tool_calls: bool,
) -> bool:
    """Detect short planning-only replies that should not end the turn.

    Observed failure mode: Claude Code returns a successful result whose entire
    body is first-thoughts / process narration (\"I'll do a fresh pass...\") and
    Hermes treats that as the final answer. When tools were available and no
    tool_call was emitted, retry instead of answering with the preamble.
    """

    if has_tool_calls:
        return False
    # Only apply when the agent turn had tools — pure chat can legitimately
    # be a short acknowledgment.
    if not had_tools:
        return False
    body = (text or "").strip()
    if not body:
        return True
    if len(body) > 350:
        return False
    # Multi-paragraph answers are real content.
    if body.count("\n\n") >= 2:
        return False
    lower = body.lower()
    starts = lower.startswith(_INCOMPLETE_PREAMBLE_STARTERS) or any(
        lower.startswith(s) for s in _INCOMPLETE_PREAMBLE_STARTERS
    )
    if not starts:
        # Also catch leading markdown bullets / fillers before the starter.
        stripped = lower.lstrip("#>*- \t")
        starts = stripped.startswith(_INCOMPLETE_PREAMBLE_STARTERS)
    if not starts:
        return False
    # Short planning sentence(s) without a substantial body.
    sentences = [s.strip() for s in body.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return len(sentences) <= 3 and len(body) <= 350


def _response_has_tool_calls(text: str) -> bool:
    body = text or ""
    return "<tool_call>" in body or "</tool_call>" in body


def _build_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child-process environment.

    Authentication is delegated to the already-authenticated Claude Code CLI
    (it reads its own OAuth/keychain state).  We never inject provider API
    keys.  Default path uses the Hermes scrubber with credentials stripped.
    There is no fail-open raw ``os.environ`` path — if the scrubber is
    unavailable, raise.
    """

    if base_env is None:
        from tools.environments.local import hermes_subprocess_env

        env = hermes_subprocess_env(inherit_credentials=False)
    else:
        env = dict(base_env)
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "C.UTF-8")
    for key in list(env):
        upper = key.upper()
        if upper.startswith("ANTHROPIC_") or upper in {
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_API_KEY",
        }:
            env.pop(key, None)
    return env


def _require_uuid_session_id(session_id: str, *, where: str) -> str:
    sid = (session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(sid):
        raise RuntimeError(
            f"Claude Code returned a malformed session_id from {where}: {sid!r}"
        )
    return sid


def _reap_process_group(process: subprocess.Popen[str], *, grace_seconds: float = 5.0) -> None:
    """Ensure the whole process group is dead after timeout/abort.

    ``process.wait()`` only reaps the leader.  After a graceful SIGTERM, kill
    the group with SIGKILL regardless of whether the leader already exited,
    then reap the leader.
    """

    try:
        process.wait(timeout=grace_seconds)
    except Exception:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except Exception:
        pass


class ClaudeCodeSession:
    """One collision-free Claude Code conversation bound to one Hermes client.

    Lifecycle
    ---------
    * First turn: no ``session_id`` -> launches ``claude`` with a fresh
      ``--session-id`` UUID, sends the complete formatted prompt over stdin,
      and records the returned ``session_id`` + message fingerprints.
    * Later turns: ``session_id`` known and prefix matches -> launches with
      ``--resume <session_id>`` and sends only the incremental new messages.
    * Prefix mismatch (``/new``, compression, transcript repair) -> starts a
      fresh session.
    * Expired/invalid server session -> retries once as a fresh conversation
      with the complete prompt.
    """

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._previous_messages: tuple[tuple[str, str], ...] = ()
        self._state_key: str | None = None
        self._bound_model: str | None = None
        self._bound_effort: str | None = None
        self._bound_tools_digest: str = ""
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._abort_requested = False
        self._request_active = False
        self._last_usage: dict[str, Any] = {}

    @property
    def last_usage(self) -> dict[str, Any]:
        """Normalized usage from the most recent successful Claude Code result."""

        with self._lock:
            return dict(self._last_usage)

    def reset(self) -> None:
        with self._lock:
            if self._state_key:
                _delete_durable_state(self._state_key)
            self._session_id = None
            self._previous_messages = ()
            self._state_key = None
            self._bound_model = None
            self._bound_effort = None
            self._bound_tools_digest = ""
            self._last_usage = {}

    def abort(self) -> None:
        """Terminate the in-flight Claude Code process group without waiting.

        Sends SIGTERM immediately, closes pipes so blocked readers unwind,
        then escalates to SIGKILL after a short grace so cancellation is not
        weaker than the timeout path — even when the CLI ignores SIGTERM.
        """

        with self._process_lock:
            if not self._request_active:
                return
            self._abort_requested = True
            process = self._active_process
        if process is None:
            return
        # Unblock readline/communicate waiters.
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

        def _escalate() -> None:
            try:
                try:
                    process.wait(timeout=0.5)
                except Exception:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            except Exception:
                pass

        threading.Thread(target=_escalate, name="claude-code-abort-escalate", daemon=True).start()

    def run(
        self,
        prompt_text: str,
        *,
        messages: list[dict[str, Any]],
        model: str,
        effort: str | None = None,
        tools_digest: str = "",
        timeout_seconds: float = 270.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        state_key: str | None = None,
        command: str | None = None,
        on_text_chunk: Any = None,
    ) -> tuple[str, str]:
        """Return ``(response, reasoning)`` using the durable Claude Code session.

        ``prompt_text`` is the *complete* formatted prompt (system + tools +
        transcript) used for a fresh session or an expired-session retry.
        ``messages`` drives incremental continuation.
        ``command`` is the resolved Claude Code CLI binary path (required for
        exact executable selection; falls back to env/PATH resolution only
        when omitted).
        """

        current = _message_fingerprint(messages)
        normalized_effort = effort.strip() if isinstance(effort, str) and effort.strip() else None
        normalized_tools = tools_digest if isinstance(tools_digest, str) else ""
        with self._lock, _durable_transition_lock(state_key):
            self._last_usage = {}
            # Always reload durable state under the lease before dispatch so a
            # long-lived owner cannot publish a divergent branch after another
            # process advanced the same conversation.
            if state_key:
                durable = _load_durable_state(state_key)
                if durable:
                    (
                        self._session_id,
                        self._previous_messages,
                        self._bound_model,
                        self._bound_effort,
                        self._bound_tools_digest,
                    ) = durable
                else:
                    # Missing/corrupt durable state invalidates warm memory for
                    # this key — never resume a stale in-memory session_id.
                    self._session_id = None
                    self._previous_messages = ()
                    self._bound_model = None
                    self._bound_effort = None
                    self._bound_tools_digest = ""
                self._state_key = state_key
            elif state_key != self._state_key:
                self._session_id = None
                self._previous_messages = ()
                self._bound_model = None
                self._bound_effort = None
                self._bound_tools_digest = ""
                self._state_key = state_key
            previous_count = len(self._previous_messages)
            identity_matches = (
                self._bound_model == model
                and self._bound_effort == normalized_effort
                and self._bound_tools_digest == normalized_tools
            )
            can_resume = bool(
                self._session_id
                and identity_matches
                and len(current) > previous_count
                and current[:previous_count] == self._previous_messages
            )
            if self._session_id and not can_resume:
                _LOG.info(
                    "Claude Code resume skipped: identity_match=%s prefix_match=%s "
                    "history_advanced=%s previous_messages=%d current_messages=%d",
                    identity_matches,
                    current[:previous_count] == self._previous_messages,
                    len(current) > previous_count,
                    previous_count,
                    len(current),
                )
            if can_resume:
                incremental = _incremental_prompt(messages, previous_count)
                if incremental:
                    try:
                        response, reasoning, session_id = self._execute_with_soft_limit_retry(
                            incremental,
                            session_id=self._session_id,
                            model=model,
                            effort=normalized_effort,
                            timeout_seconds=timeout_seconds,
                            cwd=cwd,
                            env=env,
                            command=command,
                            on_text_chunk=on_text_chunk,
                            had_tools=bool(normalized_tools),
                        )
                    except ClaudeCodeSessionExpired:
                        # Expired/invalid server session: retry once as a fresh
                        # conversation with the complete prompt.
                        self._session_id = None
                        self._previous_messages = ()
                        self._bound_model = None
                        self._bound_effort = None
                        self._bound_tools_digest = ""
                        if state_key:
                            _delete_durable_state(state_key)
                    else:
                        resolved_session_id = _require_uuid_session_id(
                            session_id or self._session_id or "",
                            where="resume",
                        )
                        self._session_id = resolved_session_id
                        self._previous_messages = current
                        self._bound_model = model
                        self._bound_effort = normalized_effort
                        self._bound_tools_digest = normalized_tools
                        if state_key:
                            _save_durable_state(
                                state_key,
                                resolved_session_id,
                                self._previous_messages,
                                model=model,
                                effort=normalized_effort,
                                tools_digest=normalized_tools,
                            )
                        return response, reasoning

            # Prompt body travels over stdin — do NOT apply argv flag-size
            # limits to it.  Only short CLI flags are size-checked in _execute.
            response, reasoning, session_id = self._execute_with_soft_limit_retry(
                prompt_text,
                session_id=None,
                model=model,
                effort=normalized_effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
                command=command,
                on_text_chunk=on_text_chunk,
                had_tools=bool(normalized_tools),
            )
            resolved_session_id = _require_uuid_session_id(
                session_id, where="fresh"
            )
            self._session_id = resolved_session_id
            self._previous_messages = current
            self._bound_model = model
            self._bound_effort = normalized_effort
            self._bound_tools_digest = normalized_tools
            if state_key:
                _save_durable_state(
                    state_key,
                    self._session_id,
                    self._previous_messages,
                    model=model,
                    effort=normalized_effort,
                    tools_digest=normalized_tools,
                )
            return response, reasoning

    def _execute_with_soft_limit_retry(
        self,
        prompt_text: str,
        *,
        session_id: str | None,
        model: str,
        effort: str | None,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
        command: str | None = None,
        on_text_chunk: Any = None,
        max_attempts: int = 3,
        had_tools: bool = False,
    ) -> tuple[str, str, str]:
        """Run one CLI request, retrying soft notices and incomplete preambles.

        Soft notices and short planning-only preambles are *not* answers.
        Retry the same payload (same resume session when provided) with short
        backoff. Only after retries exhaust raise a clear provider error for
        Hermes to surface.

        Streaming is deferred until a validated answer is confirmed so a
        banner/preamble can never become the live Discord answer.
        """

        last_notice = ""
        next_prompt = prompt_text
        next_session_id = session_id
        attempts = max(1, int(max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                response, reasoning, sid = self._execute(
                    next_prompt,
                    session_id=next_session_id,
                    model=model,
                    effort=effort,
                    timeout_seconds=timeout_seconds,
                    cwd=cwd,
                    env=env,
                    command=command,
                    # Never live-stream until the attempt is validated.
                    on_text_chunk=None,
                )
            except ClaudeCodeSoftLimitNotice as exc:
                last_notice = str(exc)
                if attempt >= attempts:
                    raise RuntimeError(
                        "Claude Code CLI returned a soft usage/limit notice "
                        f"after {attempts} attempts (not treated as an answer). "
                        f"Detail: {last_notice[:400]}"
                    ) from exc
                time.sleep(min(2.0 * attempt, 6.0))
                continue

            if _is_soft_limit_notice(response):
                last_notice = response.strip()
                if attempt >= attempts:
                    raise RuntimeError(
                        "Claude Code CLI returned a soft usage/limit notice "
                        f"after {attempts} attempts (not treated as an answer). "
                        f"Detail: {last_notice[:400]}"
                    )
                time.sleep(min(2.0 * attempt, 6.0))
                continue

            if _is_incomplete_preamble_response(
                response,
                had_tools=had_tools,
                has_tool_calls=_response_has_tool_calls(response),
            ):
                last_notice = response.strip()
                if attempt >= attempts:
                    # Last attempt: still don't treat pure preamble as a real
                    # answer when tools were expected — surface a clear error
                    # so the gateway doesn't deliver first-thoughts as final.
                    raise RuntimeError(
                        "Claude Code CLI returned only intermediate planning "
                        f"text after {attempts} attempts (not treated as an "
                        f"answer). Detail: {last_notice[:400]}"
                    )
                # Continue the session that produced the preamble. Replaying
                # the complete payload would create another paid Claude turn
                # and can duplicate work already performed by the model.
                next_session_id = _require_uuid_session_id(
                    sid, where="progress-continuation"
                )
                next_prompt = _PROGRESS_CONTINUATION_PROMPT
                continue

            # Confirmed non-notice answer — emit once for stream consumers.
            if on_text_chunk is not None and response:
                try:
                    on_text_chunk(response)
                except Exception:
                    pass
            return response, reasoning, sid

        raise RuntimeError(
            "Claude Code CLI soft usage/limit or incomplete preamble after retries. "
            f"Detail: {last_notice[:400]}"
        )

    def _execute(
        self,
        prompt_text: str,
        *,
        session_id: str | None,
        model: str,
        effort: str | None,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
        command: str | None = None,
        on_text_chunk: Any = None,
    ) -> tuple[str, str, str]:
        """Run one request with an abort latch scoped to this exact call."""

        with self._process_lock:
            if self._request_active:
                raise RuntimeError("Concurrent Claude Code request on one session")
            self._request_active = True
            self._abort_requested = False
        try:
            result = self._execute_active(
                prompt_text,
                session_id=session_id,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
                command=command,
                on_text_chunk=on_text_chunk,
            )
            with self._process_lock:
                if self._abort_requested:
                    raise RuntimeError("Claude Code request aborted")
                self._active_process = None
                self._abort_requested = False
                self._request_active = False
            return result
        finally:
            with self._process_lock:
                self._active_process = None
                self._abort_requested = False
                self._request_active = False

    def _execute_active(
        self,
        prompt_text: str,
        *,
        session_id: str | None,
        model: str,
        effort: str | None,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
        command: str | None = None,
        on_text_chunk: Any = None,
    ) -> tuple[str, str, str]:
        """Launch ``claude`` with stream-json stdin and parse the event stream live.

        Returns ``(response_text, reasoning_text, session_id)``.
        When ``on_text_chunk`` is provided it is called with each assistant
        text fragment as it arrives (live stream path).
        """

        claude_bin = (command or "").strip() or _resolve_claude_command()
        argv = [
            claude_bin,
            "-p",
            "--model",
            _validate_flag_size(str(model)),
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            # Disable ALL native Claude Code tools so every tool call remains
            # under Hermes logging, permissions, MCP, and approvals.
            "--tools",
            "",
            # Suppress ALL MCP server connectors (claude.ai Gmail/Calendar/Drive,
            # .mcp.json servers, etc.).  Without this, ``--tools ""`` only disables
            # built-in tools — MCP servers still load and their tool schemas leak
            # into the model's function-calling surface, causing the model to lose
            # access to Hermes's own prompt-injected tools mid-session.
            # ``--strict-mcp-config`` ensures ONLY ``--mcp-config`` is consulted,
            # ignoring all other MCP configuration sources.
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
        ]
        if effort:
            argv += ["--effort", _validate_flag_size(str(effort))]
        if session_id:
            argv += [
                "--resume",
                _validate_flag_size(
                    _require_uuid_session_id(session_id, where="resume-arg")
                ),
            ]
        else:
            import uuid

            argv += ["--session-id", str(uuid.uuid4())]

        process_env = _build_subprocess_env(env)
        input_payload = (
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": prompt_text},
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        with self._process_lock:
            if self._abort_requested:
                raise RuntimeError("Claude Code request aborted before launch")
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=cwd or str(Path.home()),
                    env=process_env,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Claude Code CLI not found at '{claude_bin}'. "
                    "Install Claude Code or set HERMES_CLAUDE_CODE_COMMAND."
                ) from exc
            self._active_process = process

        stdin = getattr(process, "stdin", None)
        stdout_stream = getattr(process, "stdout", None)
        use_live = callable(getattr(stdin, "write", None)) and callable(
            getattr(stdout_stream, "readline", None)
        )

        stdout = ""
        stderr = ""

        if use_live:
            stderr_chunks: list[str] = []
            stdout_lines: list[str] = []
            timed_out = threading.Event()

            def _stderr_reader() -> None:
                err = getattr(process, "stderr", None)
                if err is None:
                    return
                try:
                    for line in err:
                        stderr_chunks.append(line)
                except Exception:
                    pass

            def _timeout_watchdog() -> None:
                # Unblock a hung readline() by aborting the process group and
                # closing pipes — deadline checks alone cannot run while blocked.
                timed_out.set()
                try:
                    self.abort()
                except Exception:
                    pass

            err_thread = threading.Thread(target=_stderr_reader, daemon=True)
            err_thread.start()
            # Enforce the caller timeout even when readline() is blocked.
            # Small grace covers scheduling jitter; do not add the large batch
            # drain allowance here or hung children evade the contract.
            watchdog = threading.Timer(
                max(0.05, float(timeout_seconds) + 5.0), _timeout_watchdog
            )
            watchdog.daemon = True
            watchdog.start()
            try:
                stdin.write(input_payload)
                stdin.close()
            except Exception as exc:
                watchdog.cancel()
                self.abort()
                _reap_process_group(process, grace_seconds=2.0)
                raise RuntimeError(f"Claude Code failed writing stdin: {exc}") from exc

            try:
                while True:
                    if timed_out.is_set():
                        _reap_process_group(process, grace_seconds=5.0)
                        raise RuntimeError("Claude Code request timed out")
                    if self._abort_requested and not timed_out.is_set():
                        _reap_process_group(process, grace_seconds=2.0)
                        raise RuntimeError("Claude Code request aborted")
                    line = stdout_stream.readline()
                    if line == "":
                        break
                    stdout_lines.append(line)
                    if on_text_chunk is not None and line.strip().startswith("{"):
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            event = None
                        if isinstance(event, dict) and event.get("type") == "assistant":
                            message = event.get("message") or {}
                            if isinstance(message, dict):
                                for block in message.get("content") or []:
                                    if (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):
                                        chunk = str(block.get("text") or "")
                                        if chunk:
                                            try:
                                                on_text_chunk(chunk)
                                            except Exception:
                                                pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    self.abort()
                    _reap_process_group(process, grace_seconds=2.0)
            finally:
                watchdog.cancel()
                err_thread.join(timeout=2)
            if timed_out.is_set():
                _reap_process_group(process, grace_seconds=5.0)
                raise RuntimeError("Claude Code request timed out")
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_chunks)
        else:
            # Batch path (also used by unit-test doubles exposing communicate()).
            try:
                stdout, stderr = process.communicate(
                    input=input_payload,
                    timeout=timeout_seconds + 30,
                )
            except subprocess.TimeoutExpired:
                self.abort()
                _reap_process_group(process, grace_seconds=5.0)
                raise RuntimeError("Claude Code request timed out")

        if self._abort_requested:
            _reap_process_group(process, grace_seconds=2.0)
            raise RuntimeError("Claude Code request aborted")

        stdout = stdout or ""
        stderr = stderr or ""

        if process.returncode not in (0, None) and process.returncode != 0:
            detail_parts = []
            if stderr.strip():
                detail_parts.append(stderr.strip()[-1000:])
            if stdout.strip():
                detail_parts.append(stdout.strip()[-1000:])
            detail = "\n".join(detail_parts) if detail_parts else f"exit {process.returncode}"
            if session_id and _is_expired_session_error(detail):
                raise ClaudeCodeSessionExpired(f"Claude Code failed: {detail}")
            # Rate-limit / spend-limit notices arrive as exit 1 with
            # is_error:true and a 429 / rate_limit signal.  These are
            # transient — convert to a retryable exception so the soft-limit
            # retry handler can re-attempt instead of killing the turn.
            if _is_soft_limit_detail(detail):
                raise ClaudeCodeSoftLimitNotice(detail)
            raise RuntimeError(
                f"Claude Code failed (exit {process.returncode}): {detail}"
            )

        try:
            response, reasoning, result_session_id = _parse_stream_json_output(stdout)
            self._last_usage = _parse_stream_json_usage(stdout)
        except ClaudeCodeSessionExpired:
            raise
        except RuntimeError as exc:
            if session_id and _is_expired_session_error(str(exc)):
                raise ClaudeCodeSessionExpired(str(exc)) from exc
            raise
        return response, reasoning, result_session_id



def _parse_stream_json_output(stdout: str) -> tuple[str, str, str]:
    """Parse a Claude Code ``stream-json`` stdout into (text, reasoning, session_id).

    Requires exactly one terminal successful ``result`` event and a UUID-valid
    session_id.  Partial streams (system/assistant only) are rejected.
    """

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    init_session_id = ""
    result_session_id = ""
    result_text = ""
    saw_successful_result = False
    trailing_garbage = False
    finished = False

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if finished:
            # Any non-empty content after a terminal result is truncated/malformed.
            trailing_garbage = True
            break
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude Code stream contained malformed JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise RuntimeError("Claude Code stream contained a non-object event")

        event_type = event.get("type")

        if event_type == "system":
            sid = event.get("session_id")
            if isinstance(sid, str) and sid.strip():
                init_session_id = _require_uuid_session_id(sid, where="system/init")
            continue

        if event_type == "assistant":
            message = event.get("message") or {}
            if isinstance(message, dict):
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        chunk = str(block.get("text") or "")
                        if chunk:
                            text_parts.append(chunk)
                    elif block_type in ("thinking", "reasoning"):
                        chunk = str(block.get("thinking") or block.get("text") or "")
                        if chunk:
                            reasoning_parts.append(chunk)
            continue

        if event_type == "result":
            sid = event.get("session_id")
            if not isinstance(sid, str) or not sid.strip():
                raise RuntimeError("Claude Code result event missing session_id")
            result_session_id = _require_uuid_session_id(sid, where="result")
            if init_session_id and result_session_id != init_session_id:
                raise RuntimeError(
                    "Claude Code init/result session_id mismatch: "
                    f"{init_session_id} != {result_session_id}"
                )
            # Typed successful envelope only: is_error must be the boolean False.
            # Reject missing/wrong-type is_error and error subtypes.
            is_error = event.get("is_error")
            subtype = str(event.get("subtype") or "").strip().lower()
            errors = event.get("errors")
            if is_error is not False:
                detail = str(event.get("result") or errors or "is_error not false")[:300]
                if _is_expired_session_error(detail) or (
                    isinstance(errors, list)
                    and any(_is_expired_session_error(str(e)) for e in errors)
                ):
                    raise ClaudeCodeSessionExpired(
                        f"Claude Code result error: {detail}"
                    )
                # Rate-limit / spend-limit arrives as is_error:true — make it
                # retryable instead of a hard failure.
                if _is_soft_limit_detail(detail) or _is_soft_limit_detail(
                    json.dumps(event)
                ):
                    raise ClaudeCodeSoftLimitNotice(
                        f"Claude Code rate-limit result: {detail}"
                    )
                raise RuntimeError(
                    f"Claude Code result rejected (is_error={is_error!r}, "
                    f"subtype={subtype!r}): {detail}"
                )
            if subtype and subtype not in {"success", "result_success", ""}:
                if subtype in {"error", "failure", "failed"}:
                    detail = str(event.get("result") or "")[:300]
                    if _is_expired_session_error(detail):
                        raise ClaudeCodeSessionExpired(
                            f"Claude Code result error: {detail}"
                        )
                    raise RuntimeError(
                        f"Claude Code result subtype {subtype!r}: {detail}"
                    )
            if isinstance(errors, list) and errors:
                detail = "; ".join(str(e) for e in errors)[:300]
                if _is_expired_session_error(detail):
                    raise ClaudeCodeSessionExpired(
                        f"Claude Code result error: {detail}"
                    )
                raise RuntimeError(f"Claude Code result errors: {detail}")
            final = event.get("result")
            if not isinstance(final, str):
                raise RuntimeError(
                    "Claude Code successful result missing string result field"
                )
            result_text = final
            saw_successful_result = True
            finished = True
            continue

    if trailing_garbage:
        raise RuntimeError("Claude Code stream had content after the terminal result")
    if not saw_successful_result:
        raise RuntimeError(
            "Claude Code stream missing a terminal successful result event"
        )

    # Authoritative terminal result text wins over partial assistant chunks.
    # If both are present and disagree, prefer the terminal result (complete).
    streamed = "".join(text_parts)
    response = result_text if result_text.strip() else streamed
    reasoning = "".join(reasoning_parts)
    return response, reasoning, result_session_id


def _parse_stream_json_usage(stdout: str) -> dict[str, Any]:
    """Normalize token and cost telemetry from the terminal result event.

    Anthropic reports uncached input, cache creation, and cache reads as
    separate counters. OpenAI-compatible consumers expect ``prompt_tokens``
    to include all three and expose the cache-read subset separately.
    """

    result: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if not result or result.get("is_error") is not False:
        return {}
    raw = result.get("usage")
    if not isinstance(raw, dict):
        raw = {}

    def count(name: str) -> int:
        value = raw.get(name)
        return max(0, int(value)) if isinstance(value, (int, float)) else 0

    input_tokens = count("input_tokens")
    output_tokens = count("output_tokens")
    cache_write_tokens = count("cache_creation_input_tokens")
    cached_tokens = count("cache_read_input_tokens")
    prompt_tokens = input_tokens + cache_write_tokens + cached_tokens
    service_tier = raw.get("service_tier")
    total_cost = result.get("total_cost_usd")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "total_cost_usd": float(total_cost) if isinstance(total_cost, (int, float)) else None,
        "service_tier": service_tier if isinstance(service_tier, str) else None,
        "duration_ms": result.get("duration_ms"),
        "duration_api_ms": result.get("duration_api_ms"),
    }
