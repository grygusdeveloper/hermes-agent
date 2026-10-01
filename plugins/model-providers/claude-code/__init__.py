"""Claude's own OAuth CLI using native stream-json, never generic ACP or HTTP."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeProfile(ProviderProfile):
    supports_native_streaming = True

    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(**client_kwargs)

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                **context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Hand the resolved reasoning (``reasoning_overrides``, alias presets,
        ``/reasoning``) to ClaudeCodeClient, which maps it onto ``--effort`` /
        ``--thinking``. Disabled reasoning turns thinking off at the lowest effort."""
        if not isinstance(reasoning_config, dict):
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if reasoning_config.get("enabled") is False or effort == "none":
            return {}, {"reasoning_effort": "low", "thinking": "disabled"}
        return ({}, {"reasoning_effort": effort}) if effort else ({}, {})


register_provider(ClaudeCodeProfile(
    name="claude-code", display_name="Claude Code (native CLI)", aliases=("claude-cli",),
    api_mode="chat_completions", env_vars=(), base_url="acp://claude-code",
    auth_type="external_process", process_command="claude", process_args=(),
    supports_model_listing=False, supports_health_check=False,
    supports_vision=True, supports_vision_tool_messages=False,
    fallback_models=("opus", "sonnet", "haiku", "fable"),
    model_aliases={"opus": "opus", "sonnet": "sonnet", "haiku": "haiku", "fable": "fable"},
))
