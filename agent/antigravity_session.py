"""Incremental AGY conversation transport for Google Antigravity.

The generic ACP compatibility client flattens the full OpenAI-style message
history into one prompt on every turn.  Google's consumer Antigravity endpoint
rejects that transport at roughly 25 KiB even though Gemini's actual context
window is much larger.  This module keeps one AGY conversation per client and
sends only new non-assistant messages after the first request.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

# Linux limits each execve argument to MAX_ARG_STRLEN (normally 128 KiB).
# Leave headroom for encoding and platform variation.
INLINE_PROMPT_LIMIT_BYTES = 120_000


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
    return _validate_prompt_size("\n\n".join(parts))


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

    def reset(self) -> None:
        with self._lock:
            self._conversation_id = None
            self._previous_messages = ()

    def abort(self) -> None:
        """Terminate the in-flight AGY process without waiting on state locks."""

        with self._process_lock:
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
                _validate_prompt_size(prompt_text),
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

        with self._process_lock:
            self._active_process = process

        try:
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
        finally:
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

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
        conversation_id = str(payload.get("conversation_id") or "").strip()
        return response, reasoning, conversation_id
