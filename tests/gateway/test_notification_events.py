"""Tests for safe post-delivery gateway notification events."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.notification_events import (
    build_delivery_context,
    discord_message_url,
    emit_delivery_event,
)
from gateway.run import _attach_confirmed_stream_delivery_id
from gateway.session import Platform, SessionSource


def _source(**overrides):
    values = {
        "platform": Platform.DISCORD,
        "chat_id": "1505353433685950710",
        "chat_type": "dm",
        "user_id": "1922673",
    }
    values.update(overrides)
    return SessionSource(**values)


def test_discord_dm_message_url_targets_exact_message():
    source = _source()
    assert discord_message_url(source, "1528825062260740297") == (
        "https://discord.com/channels/@me/1505353433685950710/1528825062260740297"
    )


def test_discord_guild_thread_message_url_uses_scope_and_thread():
    source = _source(
        chat_id="111111111111111111",
        thread_id="222222222222222222",
        chat_type="thread",
        scope_id="333333333333333333",
    )
    assert discord_message_url(source, "444444444444444444") == (
        "https://discord.com/channels/333333333333333333/"
        "222222222222222222/444444444444444444"
    )


@pytest.mark.parametrize(
    "source,message_id",
    [
        (_source(chat_id="not-a-snowflake"), "1528825062260740297"),
        (_source(thread_id="../../escape"), "1528825062260740297"),
        (_source(), "bad/message"),
        (_source(platform=Platform.TELEGRAM), "1528825062260740297"),
    ],
)
def test_discord_message_url_rejects_invalid_or_other_platform(source, message_id):
    assert discord_message_url(source, message_id) is None


def test_context_is_safe_bounded_and_contains_no_raw_secret():
    secret = "sk-proj-" + "X" * 48
    context = build_delivery_context(
        source=_source(),
        message_id="1528825062260740297",
        kind="final_response",
        preview=f"Done. API_KEY={secret} " + ("z" * 500),
        session_key="discord:dm:1505353433685950710",
        session_id="session-123",
    )

    assert context["event_id"] == (
        "final_response:discord:1505353433685950710:1528825062260740297"
    )
    assert context["kind"] == "final_response"
    assert context["message_id"] == "1528825062260740297"
    assert context["message_url"].endswith("/1528825062260740297")
    assert context["session_key"] == "discord:dm:1505353433685950710"
    assert context["session_id"] == "session-123"
    assert secret not in context["preview"]
    assert len(context["preview"]) <= 240
    assert "guild_id" not in context
    assert "raw_response" not in context


@pytest.mark.asyncio
async def test_emit_delivery_event_calls_registry_once_with_safe_context():
    hooks = AsyncMock()
    context = await emit_delivery_event(
        hooks=hooks,
        event_type="approval:sent",
        source=_source(),
        message_id="1528825062260740297",
        kind="approval",
        preview="needs review",
        session_key="discord:dm:1505353433685950710",
    )

    hooks.emit.assert_awaited_once_with("approval:sent", context)
    assert context["message_url"].endswith("/1528825062260740297")


@pytest.mark.asyncio
async def test_emit_delivery_event_is_fail_soft_for_missing_or_broken_registry():
    assert await emit_delivery_event(
        hooks=None,
        event_type="message:sent",
        source=_source(),
        message_id="1528825062260740297",
        kind="final_response",
    ) is None

    hooks = AsyncMock()
    hooks.emit.side_effect = RuntimeError("hook exploded")
    context = await emit_delivery_event(
        hooks=hooks,
        event_type="message:sent",
        source=_source(),
        message_id="1528825062260740297",
        kind="final_response",
    )
    assert context is not None


def test_stream_delivery_marker_requires_confirmed_suppression_and_message_id():
    consumer = SimpleNamespace(message_id="1528825062260740297")

    confirmed = {"already_sent": True, "final_response": "done"}
    _attach_confirmed_stream_delivery_id(confirmed, consumer)
    assert confirmed["_delivery_message_id"] == "1528825062260740297"

    partial = {"final_response": "still running"}
    _attach_confirmed_stream_delivery_id(partial, consumer)
    assert "_delivery_message_id" not in partial

    no_message_id = {"already_sent": True, "final_response": "done"}
    _attach_confirmed_stream_delivery_id(
        no_message_id,
        SimpleNamespace(message_id=None),
    )
    assert "_delivery_message_id" not in no_message_id
