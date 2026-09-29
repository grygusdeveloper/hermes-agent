"""Native AGY stream-json client for the profile-scoped incremental conversation transport."""
from __future__ import annotations

from typing import Any

from agent.antigravity_session import AntigravityConversation
from agent.copilot_acp_client import (
    CopilotACPClient, _completion_to_stream_chunks, _effective_timeout,
    _extract_tool_calls_from_text, _format_messages_as_prompt,
)
from agent.portal_tags import get_conversation_context
from hermes_constants import get_real_home
from tools.environments.local import hermes_subprocess_env
from types import SimpleNamespace


class AntigravityClient(CopilotACPClient):
    """Own AGY continuity without ever launching the Copilot ACP subprocess."""

    _expected_base_url = "acp://antigravity"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._conversation = AntigravityConversation()

    def close(self) -> None:
        self._conversation.abort()
        super().close()

    def get_runtime_activity(self) -> dict[str, Any]:
        return self._conversation.get_progress_snapshot()

    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
        timeout: Any = None, tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None, stream: bool = False, **_: Any,
    ) -> Any:
        if not model or model == "copilot-acp":
            raise ValueError("Antigravity requires an explicit AGY model ID")
        history = messages or []
        prompt = _format_messages_as_prompt(history, model=model, tools=tools, tool_choice=tool_choice)
        env = hermes_subprocess_env(inherit_credentials=False)
        if home := get_real_home(env):
            env["HOME"] = home
        # A model switch must never resume another model's server conversation,
        # even when the Hermes session key and message prefix are unchanged.
        # Auxiliary requests do not own the main conversation's durable checkpoint.
        conversation_key = get_conversation_context() if tools else None
        state_key = f"{model}\0{conversation_key}" if conversation_key else None
        response, reasoning = self._conversation.run(
            prompt, messages=history, model=model, timeout_seconds=_effective_timeout(timeout),
            cwd=self._acp_cwd, env=env, state_key=state_key,
        )
        tool_calls, cleaned = _extract_tool_calls_from_text(response)
        completed_ids = {str(item.get("tool_call_id")) for item in history
                         if isinstance(item, dict) and item.get("role") == "tool" and item.get("tool_call_id")}
        if any(call.id in completed_ids for call in tool_calls):
            # The model can echo executable markup from its own server history.
            # A completed call must never be issued twice after an AGY resume.
            raise RuntimeError("AGY echoed an already completed tool call; refusing duplicate execution")
        message = SimpleNamespace(content=cleaned, tool_calls=tool_calls, reasoning=reasoning or None,
                                  reasoning_content=reasoning or None, reasoning_details=None)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            model=model,
        )
        return _completion_to_stream_chunks(completion) if stream else completion
