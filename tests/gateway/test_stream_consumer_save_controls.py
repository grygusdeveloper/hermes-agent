"""Stream-consumer tests for Discord save-controls metadata propagation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter(*, saves_responses: bool = True) -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.SUPPORTS_RESPONSE_SAVE_CONTROLS = saves_responses
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="m1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="m1")
    )
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def test_metadata_for_send_marks_final_response():
    consumer = GatewayStreamConsumer(
        adapter=_make_adapter(),
        chat_id="c",
        metadata={"save_prompt": "originating prompt"},
    )
    meta = consumer._metadata_for_send(final=True, controls=True, save_content="full")
    assert meta["final_response"] is True
    assert meta["notify"] is True
    assert meta["save_response_content"] == "full"
    assert meta["save_prompt"] == "originating prompt"


def test_metadata_for_send_notify_only_without_controls():
    consumer = GatewayStreamConsumer(adapter=_make_adapter(), chat_id="c")
    meta = consumer._metadata_for_send(final=True, controls=False)
    assert meta["notify"] is True
    assert "final_response" not in meta


@pytest.mark.asyncio
async def test_first_send_finalize_carries_full_content():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    await consumer._send_or_edit("the whole answer", finalize=True)
    meta = adapter.send.await_args.kwargs["metadata"]
    assert meta["final_response"] is True
    assert meta["save_response_content"] == "the whole answer"


@pytest.mark.asyncio
async def test_preview_send_has_no_controls():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    await consumer._send_or_edit("partial preview", finalize=False)
    meta = adapter.send.await_args.kwargs["metadata"]
    assert "final_response" not in (meta or {})


@pytest.mark.asyncio
async def test_fallback_final_controls_first_chunk_only():
    adapter = _make_adapter()
    # Return distinct ids per send so continuation logic advances.
    ids = iter(range(1, 20))
    adapter.send.side_effect = lambda **k: SimpleNamespace(
        success=True, message_id=f"m{next(ids)}"
    )
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    consumer._message_id = "old"
    consumer._last_sent_text = ""
    consumer._fallback_prefix = ""

    full = "Z" * 9000  # splits into multiple fallback chunks
    await consumer._send_fallback_final(full)

    metas = [call.kwargs["metadata"] for call in adapter.send.await_args_list]
    assert len(metas) >= 2
    # First chunk carries the controls + the complete logical response.
    assert metas[0]["final_response"] is True
    assert metas[0]["save_response_content"] == full
    # Later chunks still notify but never re-attach controls.
    for later in metas[1:]:
        assert later.get("notify") is True
        assert "final_response" not in later


@pytest.mark.asyncio
async def test_new_chunk_controls_first_only_full_content():
    adapter = _make_adapter()
    ids = iter(range(1, 20))
    adapter.send.side_effect = lambda **k: SimpleNamespace(
        success=True, message_id=f"m{next(ids)}"
    )
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    full = "the complete answer"
    await consumer._send_new_chunk("part one", None, final=True, controls=True, save_content=full)
    await consumer._send_new_chunk("part two", "m1", final=True, controls=False)
    metas = [call.kwargs["metadata"] for call in adapter.send.await_args_list]
    assert metas[0]["final_response"] is True
    assert metas[0]["save_response_content"] == full
    assert "final_response" not in metas[1]


@pytest.mark.asyncio
async def test_fresh_final_carries_full_content():
    adapter = _make_adapter()
    adapter.send.side_effect = [
        SimpleNamespace(success=True, message_id="preview"),
        SimpleNamespace(success=True, message_id="fresh"),
    ]
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="c",
        config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
    )
    await consumer._send_or_edit("hello")
    consumer._message_created_ts = 0.0
    await consumer._send_or_edit("hello world complete", finalize=True)
    meta = adapter.send.await_args.kwargs["metadata"]
    assert meta["final_response"] is True
    assert meta["save_response_content"] == "hello world complete"


@pytest.mark.asyncio
async def test_split_turn_tail_persists_whole_answer():
    """A turn split across messages puts the controls on the tail, with the WHOLE answer saved."""
    adapter = _make_adapter()
    ids = iter(range(1, 20))
    adapter.send.side_effect = lambda **k: SimpleNamespace(success=True, message_id=f"m{next(ids)}")
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    consumer._append_accumulated("HEAD " * 10 + "TAIL")
    consumer._turn_split_delivery = True  # heads already sealed by the overflow splitter
    consumer._accumulated = "TAIL"
    await consumer._send_or_edit("TAIL", finalize=True)
    meta = adapter.send.await_args.kwargs["metadata"]
    assert meta["final_response"] is True
    assert meta["save_response_content"] == "HEAD " * 10 + "TAIL"


@pytest.mark.asyncio
async def test_segment_break_preamble_has_no_controls():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c")
    await consumer._send_or_edit("Let me look that up first.", finalize=True, is_turn_final=False)
    meta = adapter.send.await_args.kwargs["metadata"] or {}
    assert "final_response" not in meta


@pytest.mark.asyncio
async def test_finalize_edit_carries_controls_but_interim_edit_keeps_session_metadata():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c", metadata={"thread_id": "t"})
    await consumer._send_or_edit("partial")
    await consumer._send_or_edit("partial more")
    interim = adapter.edit_message.await_args.kwargs["metadata"]
    assert interim == {"thread_id": "t"}
    await consumer._send_or_edit("partial more done", finalize=True)
    final = adapter.edit_message.await_args.kwargs["metadata"]
    assert final == {"thread_id": "t", "final_response": True,
                     "save_response_content": "partial more done"}


@pytest.mark.asyncio
async def test_adapters_without_save_controls_get_unchanged_final_metadata():
    """Only adapters that render save controls (Discord) get the save keys; every other
    platform's final sends and finalize edits carry exactly their usual metadata."""
    adapter = _make_adapter(saves_responses=False)
    ids = iter(range(1, 20))
    adapter.send.side_effect = lambda **k: SimpleNamespace(success=True, message_id=f"m{next(ids)}")
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c", metadata={"thread_id": "t"})

    await consumer._send_or_edit("whole answer", finalize=True)
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "t", "notify": True}

    await consumer._send_or_edit("whole answer edited", finalize=True)
    assert adapter.edit_message.await_args.kwargs["metadata"] == {"thread_id": "t"}

    consumer._message_id = "old"
    await consumer._send_fallback_final("Z" * 9000)
    for call in adapter.send.await_args_list[1:]:
        meta = call.kwargs["metadata"]
        assert "final_response" not in meta and "save_response_content" not in meta


def test_discord_adapter_declares_save_controls_and_base_does_not():
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.discord.adapter import DiscordAdapter

    assert BasePlatformAdapter.SUPPORTS_RESPONSE_SAVE_CONTROLS is False
    assert DiscordAdapter.SUPPORTS_RESPONSE_SAVE_CONTROLS is True
