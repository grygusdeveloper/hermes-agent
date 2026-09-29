"""Local AGY stream-json external-process provider (not Copilot ACP)."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class AntigravityProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.antigravity_client import AntigravityClient
        return AntigravityClient(**client_kwargs)


register_provider(AntigravityProfile(
    name="antigravity", api_mode="chat_completions", env_vars=(),
    base_url="acp://antigravity", auth_type="external_process",
    process_command="agy", process_args=(),
))
