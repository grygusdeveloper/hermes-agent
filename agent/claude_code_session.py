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
_STATE_VERSION = 2
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


def _is_expired_session_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(marker in normalized for marker in _EXPIRED_SESSION_MARKERS)


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

    def abort(self) -> None:
        """Terminate the in-flight Claude Code process group without waiting.

        Sends SIGTERM immediately, then escalates to SIGKILL after a short
        grace so cancellation is not weaker than the timeout path.
        """

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
        # Escalate: do not leave children that ignore SIGTERM until communicate
        # times out.  Best-effort non-blocking grace then SIGKILL.
        def _escalate() -> None:
            try:
                if process.poll() is None:
                    try:
                        process.wait(timeout=2.0)
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
                elif state_key != self._state_key:
                    # Key changed and no durable file: clear in-memory continuity.
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
            if can_resume:
                incremental = _incremental_prompt(messages, previous_count)
                if incremental:
                    try:
                        response, reasoning, session_id = self._execute(
                            incremental,
                            session_id=self._session_id,
                            model=model,
                            effort=normalized_effort,
                            timeout_seconds=timeout_seconds,
                            cwd=cwd,
                            env=env,
                            command=command,
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
            response, reasoning, session_id = self._execute(
                prompt_text,
                session_id=None,
                model=model,
                effort=normalized_effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
                command=command,
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
    ) -> tuple[str, str, str]:
        """Launch ``claude`` with stream-json stdin and parse the event stream.

        Returns ``(response_text, reasoning_text, session_id)``.
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
        ]
        if effort:
            argv += ["--effort", _validate_flag_size(str(effort))]
        if session_id:
            argv += [
                "--resume",
                _validate_flag_size(_require_uuid_session_id(session_id, where="resume-arg")),
            ]
        else:
            # Stable fresh UUID so the session can be resumed later.
            import uuid

            argv += ["--session-id", str(uuid.uuid4())]

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
                    argv,
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
            _reap_process_group(process, grace_seconds=5.0)
            raise RuntimeError("Claude Code request timed out")

        if self._abort_requested:
            _reap_process_group(process, grace_seconds=2.0)
            raise RuntimeError("Claude Code request aborted")

        if process.returncode != 0:
            detail = stderr.strip()[-1000:] if stderr else f"exit {process.returncode}"
            if session_id and _is_expired_session_error(detail):
                raise ClaudeCodeSessionExpired(f"Claude Code failed: {detail}")
            raise RuntimeError(f"Claude Code failed (exit {process.returncode}): {detail}")

        response, reasoning, result_session_id = _parse_stream_json_output(stdout)
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
            if event.get("is_error"):
                detail = str(event.get("result") or "")[:300]
                if _is_expired_session_error(detail):
                    raise ClaudeCodeSessionExpired(
                        f"Claude Code result error: {detail}"
                    )
                raise RuntimeError(f"Claude Code result error: {detail}")
            final = event.get("result")
            if isinstance(final, str):
                result_text = final.strip()
            saw_successful_result = True
            finished = True
            continue

    if trailing_garbage:
        raise RuntimeError("Claude Code stream had content after the terminal result")
    if not saw_successful_result:
        raise RuntimeError(
            "Claude Code stream missing a terminal successful result event"
        )

    # Prefer the accumulated streamed assistant text; fall back to the
    # ``result`` envelope's flattened text when streaming produced nothing.
    response = "".join(text_parts) or result_text
    reasoning = "".join(reasoning_parts)
    return response, reasoning, result_session_id
