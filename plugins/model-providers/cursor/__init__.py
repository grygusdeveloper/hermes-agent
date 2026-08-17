"""Cursor Agent CLI provider profile.

Cursor owns authentication and refresh state. Hermes invokes the supported
Cursor ACP stdio mode and never stores or passes Cursor credentials itself.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CursorProfile(ProviderProfile):
    """Cursor Agent — local authenticated ACP subprocess."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model discovery is performed by ``agent models`` in cursor_cli."""
        return None


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://cursor",
    auth_type="external_process",
)

register_provider(cursor)
