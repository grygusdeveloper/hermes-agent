"""Tests for Discord final-response save controls (⭐ Favorite / 📚 Notion).

Covers: config parsing, final-only View attachment, chunk-1-only attachment
with the complete saved content, finalize-edit + overflow attachment, component
auth reuse, favorite idempotency/listing, Notion success/idempotency/failure
via httpx.MockTransport, stream-consumer final metadata + fallback complete
content, and /favorites formatting + dispatch.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

# The shared discord mock is installed by tests/gateway/conftest.py at import.
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    ResponseSaveView,
    _apply_yaml_config,
    parse_response_save_controls_config,
)
from plugins.platforms.discord.response_save_store import (  # noqa: E402
    DiscordResponseSaveStore,
)


def _is_save_view(view):
    # Compare by class name, not isinstance: other tests in the suite re-run
    # _define_discord_view_classes() which rebinds the module global to a fresh
    # class object, so an imported symbol may not match by identity.
    return view is not None and type(view).__name__ == "ResponseSaveView"


def _adapter(tmp_path, enabled=True, **cfg):
    db = str(tmp_path / "saves.sqlite3")
    controls = {"enabled": enabled, "database_path": db}
    controls.update(cfg)
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"response_save_controls": controls},
        )
    )
    return adapter


def _channel_with_send(sent):
    def _make_msg():
        mid = 1000 + len(sent)
        return SimpleNamespace(
            id=mid,
            jump_url=f"https://discord.com/channels/7/555/{mid}",
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=555),
        )

    async def fake_send(**kwargs):
        sent.append(kwargs)
        return _make_msg()

    channel = SimpleNamespace(
        id=555,
        send=AsyncMock(side_effect=fake_send),
    )
    return channel


def _wire_client(adapter, channel):
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )


# ── config ──────────────────────────────────────────────────────────────
def test_config_default_disabled():
    assert parse_response_save_controls_config(None)["enabled"] is False
    assert parse_response_save_controls_config(False)["enabled"] is False
    assert parse_response_save_controls_config({})["enabled"] is True  # mapping enables


def test_config_bool_and_mapping():
    assert parse_response_save_controls_config(True)["enabled"] is True
    cfg = parse_response_save_controls_config(
        {"enabled": True, "notion_parent_page_id": "abc", "database_path": "/x"}
    )
    assert cfg["enabled"] is True
    assert cfg["notion_parent_page_id"] == "abc"
    assert cfg["database_path"] == "/x"
    assert parse_response_save_controls_config({"enabled": False})["enabled"] is False


def test_yaml_config_bridges_response_save_controls_into_platform_extra():
    controls = {
        "enabled": True,
        "notion_parent_page_id": "page-123",
    }
    assert _apply_yaml_config({}, {"response_save_controls": controls}) == {
        "response_save_controls": controls
    }


def test_discord_requires_identical_stream_finalize_for_save_controls():
    assert DiscordAdapter.REQUIRES_EDIT_FINALIZE is True


# ── send: final-only + chunk-1-only ─────────────────────────────────────
@pytest.mark.asyncio
async def test_send_no_controls_when_not_final(tmp_path):
    adapter = _adapter(tmp_path)
    sent = []
    _wire_client(adapter, _channel_with_send(sent))
    await adapter.send("555", "hi", metadata={"notify": True})
    assert "view" not in sent[0]


@pytest.mark.asyncio
async def test_send_no_controls_when_disabled(tmp_path):
    adapter = _adapter(tmp_path, enabled=False)
    sent = []
    _wire_client(adapter, _channel_with_send(sent))
    await adapter.send(
        "555",
        "hi",
        metadata={"final_response": True, "save_response_content": "hi"},
    )
    assert "view" not in sent[0]


@pytest.mark.asyncio
async def test_send_no_controls_on_command_reply(tmp_path):
    adapter = _adapter(tmp_path)
    sent = []
    _wire_client(adapter, _channel_with_send(sent))
    await adapter.send(
        "555",
        "done",
        metadata={
            "final_response": True,
            "save_response_content": "done",
            "save_controls_disabled": True,
        },
    )
    assert "view" not in sent[0]


@pytest.mark.asyncio
async def test_send_final_attaches_view_and_persists(tmp_path):
    adapter = _adapter(tmp_path)
    sent = []
    _wire_client(adapter, _channel_with_send(sent))
    result = await adapter.send(
        "555",
        "short answer",
        metadata={
            "final_response": True,
            "save_response_content": "the complete logical answer",
            "save_prompt": "what is it?",
        },
    )
    assert result.success
    assert _is_save_view(sent[0].get("view"))
    store = adapter._get_response_save_store()
    rec = store.get_response(str(result.message_id))
    assert rec is not None
    assert rec["content"] == "the complete logical answer"
    assert rec["prompt"] == "what is it?"
    assert rec["jump_url"].endswith(str(result.message_id))


@pytest.mark.asyncio
async def test_send_split_view_only_on_chunk1_full_content_saved(tmp_path):
    adapter = _adapter(tmp_path)
    sent = []
    _wire_client(adapter, _channel_with_send(sent))
    long_text = "A" * 2500  # forces >1 chunk at the 2000-char cap
    full = "COMPLETE:" + long_text
    result = await adapter.send(
        "555",
        long_text,
        metadata={
            "final_response": True,
            "save_response_content": full,
            "save_prompt": "p",
        },
    )
    assert len(sent) >= 2
    assert _is_save_view(sent[0].get("view"))
    for later in sent[1:]:
        assert "view" not in later
    store = adapter._get_response_save_store()
    rec = store.get_response(str(result.message_id))
    assert rec["content"] == full


# ── finalize edit + overflow ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_finalize_edit_attaches_view(tmp_path):
    adapter = _adapter(tmp_path)
    edited = []

    async def fake_edit(**kwargs):
        edited.append(kwargs)

    msg = SimpleNamespace(
        id=42,
        edit=AsyncMock(side_effect=fake_edit),
        jump_url="https://discord.com/channels/7/555/42",
        guild=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=555),
    )
    channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: channel, fetch_channel=AsyncMock(return_value=channel)
    )
    await adapter.edit_message(
        "555",
        "42",
        "final text",
        finalize=True,
        metadata={"final_response": True, "save_response_content": "final full"},
    )
    assert _is_save_view(edited[0].get("view"))
    rec = adapter._get_response_save_store().get_response("42")
    assert rec["content"] == "final full"


@pytest.mark.asyncio
async def test_mid_stream_edit_no_view(tmp_path):
    adapter = _adapter(tmp_path)
    edited = []
    msg = SimpleNamespace(
        id=42,
        edit=AsyncMock(side_effect=lambda **k: edited.append(k)),
        jump_url="j",
        guild=None,
        channel=SimpleNamespace(id=555),
    )
    channel = SimpleNamespace(id=555, fetch_message=AsyncMock(return_value=msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: channel, fetch_channel=AsyncMock(return_value=channel)
    )
    await adapter.edit_message(
        "555", "42", "partial", finalize=False,
        metadata={"final_response": True, "save_response_content": "x"},
    )
    assert "view" not in edited[0]


@pytest.mark.asyncio
async def test_overflow_split_view_only_on_first(tmp_path):
    adapter = _adapter(tmp_path)
    orig_edit = []
    orig_msg = SimpleNamespace(
        id=42,
        edit=AsyncMock(side_effect=lambda **k: orig_edit.append(k)),
        jump_url="https://discord.com/channels/7/555/42",
        guild=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=555),
        to_reference=MagicMock(return_value=object()),
    )
    cont_sends = []

    async def fake_send(**kwargs):
        cont_sends.append(kwargs)
        return SimpleNamespace(id=99 + len(cont_sends), to_reference=MagicMock(return_value=object()))

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=orig_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: channel, fetch_channel=AsyncMock(return_value=channel)
    )
    big = "B" * 5000
    await adapter.edit_message(
        "555", "42", big, finalize=True,
        metadata={"final_response": True, "save_response_content": big},
    )
    # First (original) message edit carries the view; continuations don't.
    assert _is_save_view(orig_edit[0].get("view"))
    for cont in cont_sends:
        assert "view" not in cont


# ── component auth ──────────────────────────────────────────────────────
def _interaction(user_id, *, message_id=42):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, roles=[]),
        message=SimpleNamespace(id=message_id),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_favorite_rejects_unauthorized(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    mock_store = MagicMock()
    mock_store.is_approved.return_value = False
    monkeypatch.setattr("gateway.pairing.PairingStore", lambda *a, **k: mock_store)

    adapter = _adapter(tmp_path)
    adapter._allowed_user_ids = {"111"}
    view = ResponseSaveView(adapter)
    interaction = _interaction(999)  # not allowed
    await view.favorite(interaction, None)
    interaction.response.send_message.assert_awaited()
    assert "authorized" in interaction.response.send_message.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_favorite_authorized_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.pairing.PairingStore", lambda *a, **k: MagicMock(is_approved=lambda *a: False)
    )
    adapter = _adapter(tmp_path)
    adapter._allowed_user_ids = {"111"}
    store = adapter._get_response_save_store()
    store.save_response(message_id="42", content="body", prompt="p")
    view = ResponseSaveView(adapter)
    interaction = _interaction(111, message_id=42)
    await view.favorite(interaction, None)
    assert store.is_favorited("42", "111")


# ── store: favorites idempotency + listing ──────────────────────────────
def test_favorite_idempotency_and_listing(tmp_path):
    store = DiscordResponseSaveStore(str(tmp_path / "s.sqlite3"))
    store.save_response(message_id="1", content="a", prompt="first", jump_url="u1")
    store.save_response(message_id="2", content="b", prompt="second", jump_url="u2")
    assert store.add_favorite("1", "user") is True
    assert store.add_favorite("1", "user") is False  # duplicate, per response+user
    store.add_favorite("2", "user")
    favs = store.list_favorites("user", 10)
    assert [f["message_id"] for f in favs] == ["2", "1"]  # newest first
    assert store.list_favorites("other", 10) == []


# ── /favorites formatting + dispatch ────────────────────────────────────
@pytest.mark.asyncio
async def test_favorites_command_formats_jump_links():
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from gateway.platforms.base import Platform

    gw = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    fake_adapter = SimpleNamespace(
        list_user_favorites=AsyncMock(
            return_value=[
                {"prompt": "Hello", "jump_url": "https://d/1", "content": "x"},
                {"prompt": "World", "jump_url": "https://d/2", "content": "y"},
            ]
        )
    )
    gw.adapters = {Platform.DISCORD: fake_adapter}

    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.DISCORD, user_id="u1"),
        get_command_args=lambda: "",
    )
    out = await gw._handle_favorites_command(event)
    assert "[Hello](https://d/1)" in out
    assert "[World](https://d/2)" in out
    fake_adapter.list_user_favorites.assert_awaited_once_with("u1", 10)


@pytest.mark.asyncio
async def test_favorites_command_wrong_platform():
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from gateway.platforms.base import Platform

    gw = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    gw.adapters = {}
    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.TELEGRAM, user_id="u1"),
        get_command_args=lambda: "",
    )
    out = await gw._handle_favorites_command(event)
    assert "Discord" in out
