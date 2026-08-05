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
_STATE_VERSION = 1
_STATE_LOCK = threading.Lock()

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
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
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
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
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
    return session_id, tuple(fingerprints)


def _save_durable_state(
    state_key: str,
    session_id: str,
    fingerprints: tuple[tuple[str, str], ...],
) -> None:
    directory = _state_dir()
    path = _state_path(state_key)
    payload = {
        "version": _STATE_VERSION,
        "state_key_hash": hashlib.sha256(state_key.encode("utf-8")).hexdigest(),
        "session_id": session_id,
        "message_fingerprints": [list(item) for item in fingerprints],
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
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "tool": "Tool Result",
        }.get(role, role.title() or "Context")
        parts.append(f"{label}:\n{rendered}")
    return "\n\n".join(parts)


class ClaudeCodeSessionExpired(RuntimeError):
    """Claude Code positively identified a missing/invalid/expired session."""


def _is_expired_session_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(marker in normalized for marker in _EXPIRED_SESSION_MARKERS)


def _build_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child-process environment.

    Authentication is delegated to the already-authenticated Claude Code CLI
    (it reads its own OAuth/keychain state).  We never inject credentials here.
    """

    env = dict(base_env or os.environ)
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "C.UTF-8")
    # Claude Code writes session history under this; keep it stable per host.
    env.pop("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", None)
    return env


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
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._abort_requested = False
        self._request_active = False

    def reset(self) -> None:
        with self._lock:
            if self._state_key:
                _delete_durable_state(self._state_key)
            self._session_id = None
            self._previous_messages = ()
            self._state_key = None

    def abort(self) -> None:
        """Terminate the in-flight Claude Code process group without waiting."""

        with self._process_lock:
            if not self._request_active:
                return
            self._abort_requested = True
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def run(
        self,
        prompt_text: str,
        *,
        messages: list[dict[str, Any]],
        model: str,
        effort: str | None = None,
        timeout_seconds: float = 270.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        state_key: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(response, reasoning)`` using the durable Claude Code session.

        ``prompt_text`` is the *complete* formatted prompt (system + tools +
        transcript) used for a fresh session or an expired-session retry.
        ``messages`` drives incremental continuation.
        """

        current = _message_fingerprint(messages)
        with self._lock, _durable_transition_lock(state_key):
            if state_key != self._state_key:
                self._session_id = None
                self._previous_messages = ()
                self._state_key = state_key
                if state_key:
                    durable = _load_durable_state(state_key)
                    if durable:
                        self._session_id, self._previous_messages = durable
            previous_count = len(self._previous_messages)
            can_resume = bool(
                self._session_id
                and len(current) > previous_count
                and current[:previous_count] == self._previous_messages
            )
            if can_resume:
                incremental = _incremental_prompt(messages, previous_count)
                if incremental:
                    try:
                        response, reasoning, session_id = self._execute(
                            incremental,
                            session_id=self._session_id,
                            model=model,
                            effort=effort,
                            timeout_seconds=timeout_seconds,
                            cwd=cwd,
                            env=env,
                        )
                    except ClaudeCodeSessionExpired:
                        # Expired/invalid server session: retry once as a fresh
                        # conversation with the complete prompt.
                        self._session_id = None
                        self._previous_messages = ()
                        if state_key:
                            _delete_durable_state(state_key)
                    else:
                        resolved_session_id = session_id or self._session_id
                        if not resolved_session_id:
                            raise RuntimeError(
                                "Claude Code resume did not preserve a session_id"
                            )
                        self._session_id = resolved_session_id
                        self._previous_messages = current
                        if state_key:
                            _save_durable_state(
                                state_key,
                                resolved_session_id,
                                self._previous_messages,
                            )
                        return response, reasoning

            # Prompt body travels over stdin — do NOT apply argv flag-size
            # limits to it.  Only short CLI flags are size-checked in _execute.
            response, reasoning, session_id = self._execute(
                prompt_text,
                session_id=None,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
            if not session_id:
                raise RuntimeError(
                    "Claude Code did not return a session_id; durable "
                    "continuation cannot be established"
                )
            self._session_id = session_id
            self._previous_messages = current
            if state_key:
                _save_durable_state(
                    state_key,
                    self._session_id,
                    self._previous_messages,
                )
            return response, reasoning

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
    ) -> tuple[str, str, str]:
        """Launch ``claude`` with stream-json stdin and parse the event stream.

        Returns ``(response_text, reasoning_text, session_id)``.
        """

        claude_bin = _resolve_claude_command()
        command = [
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
        ]
        if effort:
            command += ["--effort", _validate_flag_size(str(effort))]
        if session_id:
            command += ["--resume", _validate_flag_size(str(session_id))]
        else:
            # Stable fresh UUID so the session can be resumed later.
            import uuid

            command += ["--session-id", str(uuid.uuid4())]

        process_env = _build_subprocess_env(env)

        # stream-json input: one JSON object per line, then close stdin.
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
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
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

        try:
            stdout, stderr = process.communicate(
                input=input_payload,
                timeout=timeout_seconds + 30,
            )
        except subprocess.TimeoutExpired:
            self.abort()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()
            raise RuntimeError("Claude Code request timed out")

        if self._abort_requested:
            raise RuntimeError("Claude Code request aborted")

        if process.returncode != 0:
            detail = stderr.strip()[-1000:] if stderr else f"exit {process.returncode}"
            if session_id and _is_expired_session_error(detail):
                raise ClaudeCodeSessionExpired(f"Claude Code failed: {detail}")
            raise RuntimeError(f"Claude Code failed (exit {process.returncode}): {detail}")

        response, reasoning, result_session_id = _parse_stream_json_output(stdout)
        if not response and not reasoning:
            detail = stderr.strip()[-500:] if stderr else "empty response"
            # An empty result with a non-zero-ish stderr may still be an
            # expired-session case where the CLI exited 0 with an error event.
            if session_id and _is_expired_session_error(detail):
                raise ClaudeCodeSessionExpired(f"Claude Code empty response: {detail}")
        else:
            detail = ""
        if not result_session_id:
            # The ``result`` event always carries the session_id for a
            # successful turn.  Missing it means the stream was malformed.
            raise RuntimeError(
                f"Claude Code did not return a session_id; stderr: {detail or '(none)'}"
            )
        return response, reasoning, result_session_id


def _parse_stream_json_output(stdout: str) -> tuple[str, str, str]:
    """Parse a Claude Code ``stream-json`` stdout into (text, reasoning, session_id).

    Event types observed (Claude Code 2.1.x):
      * ``system`` (subtype ``init``) — carries the initial ``session_id``.
      * ``assistant`` — ``message.content`` may contain ``{type:text}`` and
        ``{type:thinking}`` blocks; text accumulates the response, thinking
        accumulates reasoning.
      * ``user`` — tool results (ignored here; Claude Code tools are disabled).
      * ``result`` — final envelope with ``result`` (full text), ``session_id``,
        and ``is_error``.
    """

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    session_id = ""
    result_text = ""

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")

        if event_type == "system":
            sid = event.get("session_id")
            if isinstance(sid, str) and sid.strip():
                session_id = sid.strip()
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
            if isinstance(sid, str) and sid.strip():
                session_id = sid.strip()
            if event.get("is_error"):
                detail = str(event.get("result") or "")[:300]
                if _is_expired_session_error(detail):
                    raise ClaudeCodeSessionExpired(
                        f"Claude Code result error: {detail}"
                    )
                raise RuntimeError(f"Claude Code result error: {detail}")
            final = event.get("result")
            if isinstance(final, str) and final.strip():
                result_text = final.strip()
            continue

    # Prefer the accumulated streamed assistant text; fall back to the
    # ``result`` envelope's flattened text when streaming produced nothing.
    response = "".join(text_parts) or result_text
    reasoning = "".join(reasoning_parts)
    return response, reasoning, session_id
