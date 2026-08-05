"""Incremental AGY conversation transport for Google Antigravity.

The generic ACP compatibility client flattens the full OpenAI-style message
history into one prompt on every turn.  Google's consumer Antigravity endpoint
rejects that transport at roughly 25 KiB even though Gemini's actual context
window is much larger.  This module keeps one AGY conversation per client and
sends only new non-assistant messages after the first request.
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
from pathlib import Path
from typing import Any

# Linux limits each execve argument to MAX_ARG_STRLEN (normally 128 KiB).
# Leave headroom for encoding and platform variation.
INLINE_PROMPT_LIMIT_BYTES = 120_000
# AGY's --print has no stdin or file-attachment path that reaches the
# tools-disabled Hermes agent (its "@path" expansion depends on the model's
# own native file tool, which the Hermes agent profile deliberately has none
# of, so Hermes stays the sole tool-call owner). A prompt over the argv
# ceiling is instead delivered as multiple sequential turns on one AGY
# conversation. Reserve headroom under the raw ceiling for the small
# multi-part wrapper text added to each chunk.
_CHUNK_WRAPPER_RESERVE_BYTES = 4_000
TRANSPORT_CHUNK_BUDGET_BYTES = INLINE_PROMPT_LIMIT_BYTES - _CHUNK_WRAPPER_RESERVE_BYTES
_CONVERSATION_ID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_MESSAGE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_VERSION = 1
_STATE_LOCK = threading.Lock()


def _render_content(content: Any) -> str:
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
            # Tool rows are crash-flushed before /steer is appended. AGY has
            # consumed the authenticated steer server-side, but a DB rebuild
            # replays the original durable tool row. Hash that durable prefix
            # so restart continuity is not broken by an intentionally
            # non-rewritten append-only row.
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
    # Execution profiles are scoped through a contextvar rather than by
    # mutating process-global HERMES_HOME. Resolve through the canonical
    # helper so spawned Gemini sessions cannot spill continuity state into
    # Main's profile directory.
    try:
        from hermes_constants import get_hermes_home

        hermes_home = Path(get_hermes_home())
    except Exception:
        hermes_home = Path(
            os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
        )
    return hermes_home / "state" / "antigravity-conversations"


def _state_path(state_key: str) -> Path:
    digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
    return _state_dir() / f"{digest}.json"


@contextmanager
def _durable_transition_lock(state_key: str | None):
    """Serialize one conversation's load → dispatch → publication transition."""

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
    conversation_id = payload.get("conversation_id")
    raw_fingerprints = payload.get("message_fingerprints")
    if not isinstance(conversation_id, str) or not _CONVERSATION_ID_RE.fullmatch(
        conversation_id
    ):
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
    return conversation_id, tuple(fingerprints)


def _save_durable_state(
    state_key: str,
    conversation_id: str,
    fingerprints: tuple[tuple[str, str], ...],
) -> None:
    directory = _state_dir()
    path = _state_path(state_key)
    payload = {
        "version": _STATE_VERSION,
        "state_key_hash": hashlib.sha256(state_key.encode("utf-8")).hexdigest(),
        "conversation_id": conversation_id,
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


def _validate_prompt_size(text: str, limit: int = INLINE_PROMPT_LIMIT_BYTES) -> str:
    """Reject an oversized exec argument instead of silently losing context."""

    size = len(text.encode("utf-8"))
    if size > limit:
        raise RuntimeError(
            "AGY prompt transport limit exceeded: "
            f"{size} UTF-8 bytes > {limit}; no context was sent or truncated"
        )
    return text


def _incremental_prompt(messages: list[dict[str, Any]], previous_count: int) -> str:
    parts: list[str] = []
    for message in messages[previous_count:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        # AGY already owns its prior assistant output in the server conversation.
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
    # Oversized output is delivered as multiple sequential turns by the
    # caller (see _execute_multipart); do not reject it here.
    return "\n\n".join(parts)


def _split_into_chunks(text: str, budget_bytes: int) -> list[str]:
    """Split text into UTF-8-safe pieces no larger than budget_bytes.

    Slices the exact UTF-8 byte payload into consecutive, non-overlapping
    windows, backing each window off to the nearest valid UTF-8 character
    boundary. Every byte of the input lands in exactly one chunk, in order,
    including separator bytes such as newlines: concatenating the returned
    chunks always reproduces the original text exactly.
    """

    payload = text.encode("utf-8")
    if len(payload) <= budget_bytes:
        return [text]
    chunks: list[str] = []
    start = 0
    total = len(payload)
    while start < total:
        end = min(start + budget_bytes, total)
        piece_bytes = payload[start:end]
        while piece_bytes:
            try:
                piece = piece_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                piece_bytes = piece_bytes[:-1]
        else:
            piece = ""
        if not piece:
            raise RuntimeError(
                "AGY chunk transport could not split content under the byte budget"
            )
        chunks.append(piece)
        start += len(piece_bytes)
    return chunks or [""]


class AntigravityConversationExpired(RuntimeError):
    """AGY positively identified a missing, invalid, or expired conversation."""


def _is_expired_conversation_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(
        marker in normalized
        for marker in (
            "conversation not found",
            "invalid conversation",
            "conversation has expired",
            "conversation expired",
            "unknown conversation",
        )
    )


class AntigravityConversation:
    """One collision-free AGY conversation bound to one Hermes model client."""

    def __init__(self) -> None:
        self._conversation_id: str | None = None
        self._previous_messages: tuple[tuple[str, str], ...] = ()
        self._state_key: str | None = None
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._abort_requested = False
        self._request_active = False
        # Set for the lifetime of a whole run() call so abort() can stop a
        # multi-part sequence between chunks, not only mid-subprocess.
        self._sequence_abort_requested = False

    def reset(self) -> None:
        with self._lock:
            if self._state_key:
                _delete_durable_state(self._state_key)
            self._conversation_id = None
            self._previous_messages = ()
            self._state_key = None

    def abort(self) -> None:
        """Terminate the in-flight AGY process without waiting on state locks."""

        with self._process_lock:
            self._sequence_abort_requested = True
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
        effort: str = "high",
        timeout_seconds: float = 270.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        state_key: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(response, reasoning)`` using incremental AGY context."""

        current = _message_fingerprint(messages)
        with self._lock, _durable_transition_lock(state_key):
            if state_key != self._state_key:
                self._conversation_id = None
                self._previous_messages = ()
                self._state_key = state_key
                if state_key:
                    durable = _load_durable_state(state_key)
                    if durable:
                        self._conversation_id, self._previous_messages = durable
            previous_count = len(self._previous_messages)
            can_resume = bool(
                self._conversation_id
                and len(current) > previous_count
                and current[:previous_count] == self._previous_messages
            )
            if can_resume:
                incremental = _incremental_prompt(messages, previous_count)
                if incremental:
                    try:
                        response, reasoning, conversation_id = self._deliver(
                            incremental,
                            conversation_id=self._conversation_id,
                            model=model,
                            effort=effort,
                            timeout_seconds=timeout_seconds,
                            cwd=cwd,
                            env=env,
                        )
                    except AntigravityConversationExpired:
                        # Expired/invalid server conversation: retry once as a
                        # fresh AGY conversation with the complete full prompt.
                        self._conversation_id = None
                        self._previous_messages = ()
                        if state_key:
                            _delete_durable_state(state_key)
                    else:
                        resolved_conversation_id = (
                            conversation_id or self._conversation_id
                        )
                        if not resolved_conversation_id:
                            raise RuntimeError(
                                "AGY resume did not preserve a conversation_id"
                            )
                        self._conversation_id = resolved_conversation_id
                        self._previous_messages = current
                        if state_key:
                            _save_durable_state(
                                state_key,
                                resolved_conversation_id,
                                self._previous_messages,
                            )
                        return response, reasoning

            response, reasoning, conversation_id = self._deliver(
                prompt_text,
                conversation_id=None,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
            if not conversation_id:
                raise RuntimeError(
                    "AGY did not return a conversation_id; incremental context cannot be established"
                )
            self._conversation_id = conversation_id
            self._previous_messages = current
            if state_key:
                _save_durable_state(
                    state_key,
                    self._conversation_id,
                    self._previous_messages,
                )
            return response, reasoning

    def _deliver(
        self,
        prompt_text: str,
        *,
        conversation_id: str | None,
        model: str,
        effort: str,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[str, str, str]:
        """Send prompt_text, splitting into sequential turns above the argv ceiling."""

        with self._process_lock:
            self._sequence_abort_requested = False
        if len(prompt_text.encode("utf-8")) <= INLINE_PROMPT_LIMIT_BYTES:
            return self._execute(
                prompt_text,
                conversation_id=conversation_id,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
        return self._execute_multipart(
            prompt_text,
            conversation_id=conversation_id,
            model=model,
            effort=effort,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
            env=env,
        )

    def _execute_multipart(
        self,
        prompt_text: str,
        *,
        conversation_id: str | None,
        model: str,
        effort: str,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[str, str, str]:
        """Deliver a body over the argv ceiling as sequential turns on one
        AGY conversation, using the same server-side resume AGY already
        provides for incremental context. AGY's --print has no stdin or file
        path that reaches the tools-disabled Hermes agent, so this is the
        only transport that neither truncates content nor writes it to disk.
        """

        # Start each multi-part sequence with a clean between-chunk latch so a
        # prior aborted sequence cannot poison this call.  In-flight aborts are
        # still caught per-chunk by _execute's own _abort_requested latch; this
        # flag only gates travel between chunks.
        with self._process_lock:
            self._sequence_abort_requested = False

        parts = _split_into_chunks(prompt_text, TRANSPORT_CHUNK_BUDGET_BYTES)
        total = len(parts)
        current_conversation_id = conversation_id
        response = ""
        reasoning = ""
        for index, part in enumerate(parts, start=1):
            with self._process_lock:
                if self._sequence_abort_requested:
                    raise RuntimeError("AGY request aborted")
            if index < total:
                wrapped = (
                    f"System: This is part {index} of {total} of one oversized "
                    "Hermes request, split only because of a local transport "
                    "limit (not a model context limit). Wait for every part "
                    "before responding to the request itself. Reply with only "
                    "the single word OK to confirm receipt of this part.\n\n" + part
                )
            else:
                wrapped = (
                    f"System: This is the final part {index} of {total}. You "
                    "now have the complete request across all parts, in order. "
                    "Process it as one message and respond normally now.\n\n" + part
                )
            response, reasoning, returned_id = self._execute(
                wrapped,
                conversation_id=current_conversation_id,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
            if not returned_id:
                raise RuntimeError(
                    "AGY did not return a conversation_id mid multi-part transport"
                )
            current_conversation_id = returned_id
        if current_conversation_id is None:
            raise RuntimeError(
                "AGY multi-part transport produced no conversation_id"
            )
        return response, reasoning, current_conversation_id

    def _execute(
        self,
        prompt_text: str,
        *,
        conversation_id: str | None,
        model: str,
        effort: str,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[str, str, str]:
        """Run one request with an abort latch scoped to this exact call."""

        with self._process_lock:
            if self._request_active:
                raise RuntimeError("Concurrent AGY request on one conversation")
            self._request_active = True
            self._abort_requested = False
        try:
            result = self._execute_active(
                prompt_text,
                conversation_id=conversation_id,
                model=model,
                effort=effort,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
            with self._process_lock:
                if self._abort_requested:
                    raise RuntimeError("AGY request aborted")
                # Success acceptance and request deactivation are one atomic
                # transition.  An abort before this lock fails this request;
                # an abort after it observes an idle conversation and is ignored.
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
        conversation_id: str | None,
        model: str,
        effort: str,
        timeout_seconds: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[str, str, str]:
        agy = os.environ.get("AGY_PATH", str(Path.home() / ".local" / "bin" / "agy"))
        command = [
            agy,
            "--agent",
            "hermes-antigravity-acp",
            "--model",
            model,
            "--effort",
            effort,
            "--sandbox",
            "--disable-slash-commands",
            "--output-format",
            "json",
            "--print-timeout",
            f"{max(1, int(timeout_seconds))}s",
        ]
        if conversation_id:
            command.extend(("--conversation", conversation_id))
        command.extend(("--print", prompt_text))

        process_env = dict(env or os.environ)
        process_env.setdefault("HOME", str(Path.home()))
        process_env.setdefault("LANG", "C.UTF-8")
        with self._process_lock:
            if self._abort_requested:
                raise RuntimeError("AGY request aborted before launch")
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or str(Path.home()),
                    env=process_env,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(f"AGY executable not found at {agy}") from exc
            self._active_process = process

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds + 30)
        except subprocess.TimeoutExpired as exc:
            self.abort()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()
            raise RuntimeError("AGY prompt timed out") from exc

        if process.returncode != 0:
            detail = stderr.strip()[-1000:] if stderr else f"exit {process.returncode}"
            if conversation_id and _is_expired_conversation_error(detail):
                raise AntigravityConversationExpired(f"AGY failed: {detail}")
            raise RuntimeError(f"AGY failed: {detail}")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AGY returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("AGY returned non-object JSON")
        if str(payload.get("status") or "") != "SUCCESS":
            detail = str(payload.get("response") or "")[:300]
            if conversation_id and _is_expired_conversation_error(detail):
                raise AntigravityConversationExpired(
                    f"AGY returned status {payload.get('status')!r}: {detail}"
                )
            raise RuntimeError(
                f"AGY returned status {payload.get('status')!r}: {detail}"
            )
        response = str(payload.get("response") or "").strip()
        if not response:
            detail = stderr.strip()[-500:] if stderr else "empty response"
            raise RuntimeError(f"AGY returned no response: {detail}")
        reasoning = str(
            payload.get("reasoning") or payload.get("thinking") or ""
        ).strip()
        raw_conversation_id = payload.get("conversation_id")
        if not isinstance(raw_conversation_id, str):
            raise RuntimeError("AGY returned a non-string conversation_id")
        conversation_id = raw_conversation_id.strip()
        if not _CONVERSATION_ID_RE.fullmatch(conversation_id):
            raise RuntimeError("AGY returned a malformed conversation_id")
        return response, reasoning, conversation_id
