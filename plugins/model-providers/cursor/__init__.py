"""VPS Cursor Agent provider: local ACP subprocess, with Cursor-owned credentials."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CursorProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.cursor_acp_client import CursorACPClient
        return CursorACPClient(**client_kwargs)


register_provider(CursorProfile(
    name="cursor", aliases=("cursor-agent", "cursor-cli"),
    api_mode="chat_completions", env_vars=(), base_url="acp://cursor",
    auth_type="external_process", process_command="agent", process_args=("acp",),
    supports_model_listing=False, supports_health_check=False,
))
