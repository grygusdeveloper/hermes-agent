"""The VPS Discord /heart command keeps topic tags and routes through the receiving bot."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.commands import resolve_command
from hermes_cli.commands_platforms import telegram_bot_commands, slack_native_slashes
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_heart_round_trip_preserves_unrelated_forum_tags():
    heart = SimpleNamespace(id=1, name="❤️")
    other = SimpleNamespace(id=2, name="research")
    parent = SimpleNamespace(available_tags=[heart, other])
    thread = SimpleNamespace(name="Research", parent=parent, applied_tags=[other])

    async def edit(**kwargs):
        thread.name = kwargs.get("name", thread.name)
        thread.applied_tags = kwargs.get("applied_tags", thread.applied_tags)

    thread.edit = AsyncMock(side_effect=edit)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="unused"))
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread)
    first = await adapter.toggle_forum_thread_heart("123")
    second = await adapter.toggle_forum_thread_heart("123")
    assert first == {"hearted": True, "title": "❤️ Research"}
    assert second == {"hearted": False, "title": "Research"}
    assert thread.applied_tags == [other]
    assert thread.edit.await_count == 2


@pytest.mark.asyncio
async def test_heart_routes_to_receiving_bot_and_stays_off_other_native_menus():
    assert resolve_command("fav") is resolve_command("heart")
    command = resolve_command("heart")
    assert command is not None and command.busy_policy == "dispatch"
    assert "heart" not in {name for name, _ in telegram_bot_commands(include_plugins=False)}
    assert "heart" not in {name for name, *_ in slack_native_slashes()}
    receiving = SimpleNamespace(toggle_forum_thread_heart=AsyncMock(
        return_value={"hearted": True, "title": "❤️ Research"}))
    source = SimpleNamespace(platform=Platform.DISCORD, thread_id="123", profile="hekate")
    handler = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    setattr(handler, "_delivery_adapter_for", lambda observed: receiving if observed is source else None)
    result = await handler._handle_heart_command(cast(MessageEvent, SimpleNamespace(source=source)))
    assert "❤️ Research" in result
    receiving.toggle_forum_thread_heart.assert_awaited_once_with("123")
