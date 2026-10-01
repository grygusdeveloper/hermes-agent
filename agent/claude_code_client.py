"""OpenAI-shaped completions and incremental streams over the OAuth-owned Claude CLI.

Port provenance: clean hermes-main-fork Claude client/session at 9992a59bb6;
introduction 2a8696c42e. Native system/transcript/resume semantics are retained.
The initial gate uses structured SDK MCP instead of parsing executable prose.
Resolved user images use native content blocks; warm final-turn reuse is separate.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from agent.claude_code_protocol import MARKER, MODEL, tool_specs
from agent.claude_code_session import ClaudeCodeSession
from agent.portal_tags import get_native_transport_identity
from agent.reasoning_effort import clamp_effort

# ``--effort`` / ``--thinking`` choices of Claude Code 2.1.x. Hermes levels outside them clamp to the
# nearest weaker one (minimal -> low, ultra -> max). "ultracode" is xhigh plus Claude Code's own workflow
# orchestration, whose native tools are off here (``--tools ""``), so it sends xhigh.
CLI_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_EFFORT_OVERRIDES = {"ultracode": "xhigh"}
THINKING_MODES = ("enabled", "adaptive", "disabled")


def cli_reasoning(effort: str | None, thinking: str | None = None) -> tuple[str | None, str | None]:
    """``(--effort, --thinking)`` for a requested Hermes effort and thinking mode.

    ``none`` (reasoning off) turns thinking off at the lowest effort; unknown values raise.
    """
    requested = str(effort or "").strip().lower() or None
    mode = str(thinking or "").strip().lower() or None
    if mode is not None and mode not in THINKING_MODES:
        raise ValueError("Unsupported Claude native thinking mode")
    if requested == "none" or mode == "disabled":
        return "low", "disabled"
    if requested is None:
        return None, mode
    clamped = clamp_effort(requested, CLI_EFFORTS, _EFFORT_OVERRIDES)
    if clamped not in CLI_EFFORTS:
        raise ValueError("Unsupported Claude native reasoning effort")
    return clamped, mode


class ClaudeCodeClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, base_url: str | None = None, command: str | None = None,
                 acp_command: str | None = None, args: list | None = None, acp_args: list | None = None,
                 cwd: str | None = None, acp_cwd: str | None = None, timeout: Any = None,
                 auxiliary: bool = False, **_: Any):
        if base_url and base_url.rstrip("/") != MARKER:
            raise ValueError("Claude native requires its matching transport marker; HTTP and ACP fallback are forbidden")
        if args or acp_args:
            raise ValueError("Claude native owns stream-json argv; custom ACP arguments are unsupported")
        self.base_url = MARKER
        self.timeout = timeout
        self._session = ClaudeCodeSession(command or acp_command or "claude", cwd or acp_cwd)
        self._auxiliary = auxiliary
        self._stream = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    def cancel(self) -> None:
        self._session.cancel()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._session.close()

    def _create_chat_completion(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                                tool_choice: Any = None, timeout: Any = None, stream: bool = False,
                                reasoning_effort: str | None = None, thinking: str | None = None, **_: Any):
        if not MODEL.fullmatch(model or ""):
            raise ValueError("Claude native requires a Claude model or opus/sonnet/haiku/fable alias")
        if self._stream is not None and not self._stream.finished:
            raise RuntimeError("Claude native stream must be exhausted or closed before the next request")
        effort, thinking = cli_reasoning(reasoning_effort, thinking)
        seconds = timeout if timeout is not None else self.timeout
        if seconds is None:
            seconds = 900.0
        if not isinstance(seconds, (int, float)):
            seconds = getattr(seconds, "read", None)
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Claude native requires a positive request timeout")
        specs = tool_specs(tools, tool_choice)
        # A main conversation may intentionally have no tools. Aux factories
        # explicitly opt out of durable turn state inside the same context.
        owner = None if self._auxiliary else get_native_transport_identity()
        kwargs: dict[str, Any] = dict(model=model, messages=messages, specs=specs, owner=owner,
                      timeout=float(seconds), effort=effort,
                      required=tool_choice == "required" or isinstance(tool_choice, dict))
        if thinking:
            kwargs["thinking"] = thinking
        if stream:
            self._stream = self._session.stream(**kwargs)
            return self._stream
        return self._session.run(**kwargs)
