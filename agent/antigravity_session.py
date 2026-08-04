"""Incremental AGY conversation transport for Google Antigravity.

The generic ACP compatibility client flattens the full OpenAI-style message
history into one prompt on every turn.  Google's consumer Antigravity endpoint
rejects that transport at roughly 25 KiB even though Gemini's actual context
window is much larger.  This module keeps one AGY conversation per client and
sends only new non-assistant messages after the first request.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Linux limits each execve argument to MAX_ARG_STRLEN (normally 128 KiB).
# Leave headroom for encoding and platform variation.
INLINE_PROMPT_LIMIT_BYTES = 120_000
SPOOL_PROMPT_LIMIT_BYTES = 1_000_000
_DEFAULT_AGENT = "hermes-antigravity-acp"
_SPOOL_AGENT = "hermes-antigravity-acp-spool"
_CONVERSATION_ID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


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


def _message_fingerprint(messages: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Return a deterministic, non-secret in-memory prefix identity."""

    return tuple(
        (
            str(message.get("role") or "").lower(),
            _render_content(message.get("content")),
        )
        for message in messages
        if isinstance(message, dict)
    )


def _validate_prompt_size(text: str, limit: int = INLINE_PROMPT_LIMIT_BYTES) -> str:
    """Reject an oversized exec argument instead of silently losing context."""

    size = len(text.encode("utf-8"))
    if size > limit:
        raise RuntimeError(
            "AGY prompt transport limit exceeded: "
            f"{size} UTF-8 bytes > {limit}; no context was sent or truncated"
        )
    return text


@contextmanager
def _private_prompt_spool(text: str):
    """Yield a private, exact-byte prompt file and remove it deterministically."""

    payload = text.encode("utf-8")
    if len(payload) > SPOOL_PROMPT_LIMIT_BYTES:
        raise RuntimeError(
            "AGY spool transport limit exceeded: "
            f"{len(payload)} UTF-8 bytes > {SPOOL_PROMPT_LIMIT_BYTES}; "
            "no context was sent or truncated"
        )
    with tempfile.TemporaryDirectory(prefix="hermes-antigravity-spool-") as directory:
        os.chmod(directory, 0o700)
        path = Path(directory) / "canonical-prompt.txt"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        actual = path.read_bytes()
        if actual != payload:
            raise RuntimeError("AGY spool prompt failed exact-byte verification")
        yield path, len(payload), hashlib.sha256(payload).hexdigest()


def _parse_spool_stream(
    stdout: str,
    *,
    expected_path: Path,
    expected_bytes: int,
    expected_model: str,
) -> dict[str, Any]:
    """Validate AGY's spool-loader trace and return its terminal result."""

    result: dict[str, Any] | None = None
    conversation_id: str | None = None
    completed_reads = 0
    tool_events = 0
    initialized = False
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AGY returned malformed stream JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("AGY returned a non-object stream event")
        if event.get("event") == "init":
            if initialized:
                raise RuntimeError("AGY spool stream repeated its init event")
            initialized = True
            raw_id = event.get("conversation_id")
            if isinstance(raw_id, str):
                conversation_id = raw_id.strip()
            init = event.get("init") or {}
            if not isinstance(init, dict) or init.get("agent") != _SPOOL_AGENT:
                raise RuntimeError("AGY spool trace initialized the wrong agent")
            if init.get("model") != expected_model:
                raise RuntimeError("AGY spool trace initialized the wrong model")
        update = event.get("step_update") or {}
        if isinstance(update, dict) and update.get("step_type") == "tool":
            tool_events += 1
            if tool_events > 32:
                raise RuntimeError("AGY spool loader exceeded the tool-event limit")
            info = update.get("tool_info") or {}
            if not isinstance(info, dict):
                raise RuntimeError("AGY spool trace omitted tool metadata")
            name = str(update.get("tool_name") or info.get("name") or "")
            parameters = info.get("parameters") or {}
            if name != "view_file" or not isinstance(parameters, dict):
                raise RuntimeError(f"AGY spool loader used forbidden tool {name!r}")
            if parameters.get("AbsolutePath") != str(expected_path):
                raise RuntimeError("AGY spool loader accessed an unexpected path")
            if str(update.get("state") or "") == "DONE":
                if info.get("error"):
                    raise RuntimeError("AGY spool view_file returned an error")
                output = str(info.get("output") or "")
                if f"{expected_bytes} bytes" not in output:
                    raise RuntimeError("AGY spool view_file did not attest the exact byte count")
                completed_reads += 1
        if event.get("event") == "result":
            candidate = event.get("result")
            if not isinstance(candidate, dict):
                raise RuntimeError("AGY spool stream omitted its result object")
            result = dict(candidate)
    if not initialized:
        raise RuntimeError("AGY spool stream had no init event")
    if completed_reads < 1:
        raise RuntimeError("AGY spool loader did not complete an exact-path read")
    if result is None:
        raise RuntimeError("AGY spool stream had no terminal result")
    if conversation_id and not result.get("conversation_id"):
        result["conversation_id"] = conversation_id
    return result


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
    return "\n\n".join(parts)


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
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._abort_requested = False
        self._request_active = False

    def reset(self) -> None:
        with self._lock:
            self._conversation_id = None
            self._previous_messages = ()

    def abort(self) -> None:
        """Terminate the in-flight AGY process without waiting on state locks."""

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
        effort: str = "high",
        timeout_seconds: float = 270.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Return ``(response, reasoning)`` using incremental AGY context."""

        current = _message_fingerprint(messages)
        with self._lock:
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
                        response, reasoning, conversation_id = self._execute(
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
                    else:
                        self._conversation_id = conversation_id or self._conversation_id
                        self._previous_messages = current
                        return response, reasoning

            response, reasoning, conversation_id = self._execute(
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
            return response, reasoning

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
        _spool_path: Path | None = None,
        _spool_bytes: int | None = None,
    ) -> tuple[str, str, str]:
        prompt_size = len(prompt_text.encode("utf-8"))
        if prompt_size > INLINE_PROMPT_LIMIT_BYTES and _spool_path is None:
            with _private_prompt_spool(prompt_text) as (spool_path, spool_bytes, digest):
                bootstrap = (
                    "Load the complete canonical Hermes request from absolute path "
                    f"{spool_path}. Expected UTF-8 bytes: {spool_bytes}. "
                    f"Expected SHA-256: {digest}. Process it now."
                )
                _validate_prompt_size(bootstrap)
                return self._execute_active(
                    bootstrap,
                    conversation_id=conversation_id,
                    model=model,
                    effort=effort,
                    timeout_seconds=timeout_seconds,
                    cwd=cwd,
                    env=env,
                    _spool_path=spool_path,
                    _spool_bytes=spool_bytes,
                )
        _validate_prompt_size(prompt_text)
        use_spool = _spool_path is not None
        agy = os.environ.get("AGY_PATH", str(Path.home() / ".local" / "bin" / "agy"))
        command = [
            agy,
            "--agent",
            _SPOOL_AGENT if use_spool else _DEFAULT_AGENT,
            "--model",
            model,
            "--effort",
            effort,
            "--sandbox",
            "--disable-slash-commands",
            "--output-format",
            "stream-json" if use_spool else "json",
            "--print-timeout",
            f"{max(1, int(timeout_seconds))}s",
        ]
        if conversation_id:
            command.extend(("--conversation", conversation_id))
        command.extend(("--print", prompt_text))

        process_env = dict(env or os.environ)
        process_env.setdefault("HOME", str(Path.home()))
        process_env.setdefault("LANG", "C.UTF-8")
        process_cwd = cwd or str(Path.home())
        if use_spool:
            assert _spool_path is not None
            process_cwd = str(_spool_path.parent)
            process_env["TMPDIR"] = process_cwd
            process_env["TMP"] = process_cwd
            process_env["TEMP"] = process_cwd
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
                    cwd=process_cwd,
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
        if use_spool:
            assert _spool_path is not None and _spool_bytes is not None
            payload = _parse_spool_stream(
                stdout,
                expected_path=_spool_path,
                expected_bytes=_spool_bytes,
                expected_model=model,
            )
        else:
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
