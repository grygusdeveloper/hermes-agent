"""OpenAI-compatible shim that forwards Hermes requests to the Claude Code CLI.

This adapter lets Hermes treat the locally-installed, already-authenticated
``claude`` (Claude Code CLI 2.x) as a chat-style backend — analogous to
``agent/copilot_acp_client.py`` (the GitHub Copilot / Antigravity ACP bridge)
but using Claude Code's **native stream-json stdin transport** instead of an
argv-bound prompt.

Key differences from the Copilot ACP bridge
-------------------------------------------
* **stream-json stdin** — the complete canonical Hermes request (system
  prompt, tool schemas, full transcript) travels as a JSON user-message object
  on stdin, never as an ``execve`` argument.  There is no ``MAX_ARG_STRLEN``
  ceiling.
* **Durable session continuation** — one Claude Code ``session_id`` per cached
  Hermes model client; later turns resume it with ``--resume`` and send only
  the incremental new messages.
* **Distinct identity** — provider ``claude-code``, base marker
  ``acp://claude-code``.  Does not reuse, relabel, or replace ``copilot-acp``
  or the Antigravity command.
* **Native tools disabled** — ``--tools ""`` turns off all Claude Code built-in
  tools so every tool call remains under Hermes logging, permissions, MCP, and
  approvals.  Hermes injects its tool schemas into the prompt and parses
  ``<tool_call>`` blocks back out of the response (same proven parser as the
  Copilot ACP bridge).

Security
--------
Authentication is delegated entirely to the already-authenticated Claude Code
CLI.  This module never reads, passes, or logs credentials.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.claude_code_session import ClaudeCodeSession
from agent.portal_tags import get_conversation_context

CLAUDE_CODE_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0

# Reuse the exact same proven tool-call text extraction as the Copilot ACP
# bridge.  Claude Code emits the same ``<tool_call>{...}</tool_call>`` blocks
# when its native tools are disabled and Hermes's tool schemas are injected.
from agent.copilot_acp_client import (  # noqa: E402
    _completion_to_stream_chunks,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
)


class _ClaudeCodeChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCodeChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _ClaudeCodeChatCompletions(client)


class ClaudeCodeClient:
    """Minimal OpenAI-client-compatible facade for the Claude Code CLI.

    Mirrors the shape of ``CopilotACPClient`` so the rest of Hermes
    (conversation loop, streaming, auxiliary client) can treat it uniformly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code"
        self.base_url = base_url or CLAUDE_CODE_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        # ``command``/``args`` are accepted for interface parity with
        # CopilotACPClient; only the command path is honoured (Claude Code's
        # transport is fixed: stream-json in/out, tools disabled).
        self._claude_command = command or _resolve_command()
        self._claude_cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _ClaudeCodeChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        # One durable Claude Code session per cached Hermes model client, so
        # keeping the conversation here prevents cross-thread collisions even
        # when two spawned workspaces begin with identical prompts.
        self._claude_session = ClaudeCodeSession()
        self._owns_claude_session = True

    def close(self) -> None:
        if self._owns_claude_session:
            self._claude_session.abort()
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        # Resolve the configured effort for this agent/model, if any.
        effort = _resolve_effort()

        # Only the main tool-enabled agent owns durable Claude Code continuity.
        # Auxiliary title/compression calls have no tool schema and must not
        # overwrite the main session's conversation mapping.
        state_key = get_conversation_context() if tools else None

        response_text, reasoning_text = self._claude_session.run(
            prompt_text,
            messages=messages or [],
            model=model or "sonnet",
            effort=effort,
            timeout_seconds=_effective_timeout,
            cwd=self._claude_cwd,
            env=_build_subprocess_env(),
            state_key=state_key,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-code",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_PATH", "").strip()
        or os.getenv("CLAUDE_BIN", "").strip()
        or "claude"
    )


def _resolve_effort() -> str | None:
    """Resolve the configured reasoning effort for the active agent, if any.

    Reads the agent's configured ``reasoning_effort`` without importing the
    full agent runtime (keeps this module import-light).
    """

    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        agent_cfg = config.get("agent") or {}
        if isinstance(agent_cfg, dict):
            effort = agent_cfg.get("reasoning_effort")
            if isinstance(effort, str) and effort.strip():
                normalized = effort.strip().lower()
                # Claude Code accepts: low, medium, high, xhigh, max.
                if normalized in {"low", "medium", "high", "xhigh", "max"}:
                    return normalized
    except Exception:
        pass
    return None


def _build_subprocess_env() -> dict[str, str]:
    """Build the child-process environment.

    Authentication is delegated to the already-authenticated Claude Code CLI.
    Route through the central helper so Tier-1 secrets are stripped while LLM
    provider credentials the CLI may need are preserved.
    """

    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=True)
    home = os.environ.get("HOME", "").strip()
    if home:
        env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    return env
