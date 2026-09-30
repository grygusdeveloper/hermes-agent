"""Synchronous OpenAI-shaped client over the OAuth-owned Claude CLI.

Port provenance: clean hermes-main-fork Claude client/session at 9992a59bb6;
introduction 2a8696c42e. Native system/transcript/resume semantics are retained.
The initial gate uses structured SDK MCP instead of parsing executable prose.
Streaming, vision and warm final-turn reuse are intentionally not advertised.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from agent.claude_code_protocol import MARKER, MODEL, tool_specs
from agent.claude_code_session import ClaudeCodeSession
from agent.portal_tags import get_native_transport_identity


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
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    def cancel(self) -> None:
        self._session.cancel()

    def close(self) -> None:
        self._session.close()

    def _create_chat_completion(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                                tool_choice: Any = None, timeout: Any = None, stream: bool = False,
                                reasoning_effort: str | None = None, **_: Any):
        if not MODEL.fullmatch(model or ""):
            raise ValueError("Claude native requires a Claude model or opus/sonnet/haiku/fable alias")
        if stream:
            raise ValueError("Claude native initial gate is synchronous; incremental streaming is pending")
        if reasoning_effort not in (None, "low", "medium", "high", "max"):
            raise ValueError("Unsupported Claude native reasoning effort")
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
        return self._session.run(model=model, messages=messages, specs=specs, owner=owner,
                                 timeout=float(seconds), effort=reasoning_effort,
                                 required=tool_choice == "required" or isinstance(tool_choice, dict))
