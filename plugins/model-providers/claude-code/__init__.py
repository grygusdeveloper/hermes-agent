"""Claude's own OAuth CLI using native stream-json, never generic ACP or HTTP."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeProfile(ProviderProfile):
    supports_native_streaming = True

    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(**client_kwargs)


register_provider(ClaudeCodeProfile(
    name="claude-code", display_name="Claude Code (native CLI)", aliases=("claude-cli",),
    api_mode="chat_completions", env_vars=(), base_url="acp://claude-code",
    auth_type="external_process", process_command="claude", process_args=(),
    supports_model_listing=False, supports_health_check=False,
    fallback_models=("opus", "sonnet", "haiku", "fable"),
    model_aliases={"opus": "opus", "sonnet": "sonnet", "haiku": "haiku", "fable": "fable"},
))
