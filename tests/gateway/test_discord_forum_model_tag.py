"""Model-tag replacement preserves other tags and refuses ambiguous forum state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_model_tag_replacement_preserves_other_tags_and_is_idempotent():
    old = SimpleNamespace(id=1, name="sol")
    new = SimpleNamespace(id=2, name="gemini")
    heart = SimpleNamespace(id=3, name="❤️")
    parent = SimpleNamespace(available_tags=[old, new, heart])
    thread = SimpleNamespace(parent=parent, applied_tags=[old, heart], edit=AsyncMock())

    async def edit(**kwargs):
        thread.applied_tags = kwargs["applied_tags"]

    thread.edit.side_effect = edit
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="unused"))
    adapter._client = SimpleNamespace(fetch_channel=AsyncMock(return_value=thread))
    assert await adapter.set_forum_thread_model_tag("123", "gemini", model_aliases={"sol", "gemini"})
    assert [tag.id for tag in thread.applied_tags] == [heart.id, new.id]
    assert await adapter.set_forum_thread_model_tag("123", "gemini", model_aliases={"sol", "gemini"})
    thread.edit.assert_awaited_once()
    assert not await adapter.set_forum_thread_model_tag("123", "unknown", model_aliases={"sol", "gemini"})
    thread.edit.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_tag_refuses_ambiguous_duplicate_names_without_edit():
    twins = [SimpleNamespace(id=i, name="sol") for i in (1, 2)]
    thread = SimpleNamespace(parent=SimpleNamespace(available_tags=twins),
                             applied_tags=twins[:1], edit=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="unused"))
    adapter._client = SimpleNamespace(fetch_channel=AsyncMock(return_value=thread))
    assert not await adapter.set_forum_thread_model_tag("123", "sol", model_aliases={"sol"})
    thread.edit.assert_not_awaited()
