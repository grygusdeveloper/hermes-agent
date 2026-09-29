"""Native forum creation binds brokered credentials before the first model turn."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.spawn_forum import bind_forum_post


@pytest.mark.asyncio
async def test_forum_tag_binds_model_and_execution_durably_across_homes(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "researcher").mkdir(parents=True)
    (home / "profiles" / "researcher" / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store), _session_model_overrides={})
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-1", chat_type="thread",
                           thread_id="thread-1", parent_chat_id="forum-1")
    channel = SimpleNamespace(applied_tags=[SimpleNamespace(name="sol")])
    event = MessageEvent(text="task", message_type=MessageType.TEXT, source=source,
                         raw_message=SimpleNamespace(channel=channel))
    config = {"gateway": {"spawn": {"parent_channel_id": "forum-1", "default_agent": "research",
                                 "agents": {"research": "researcher"},
                                 "models": {"sol": {"model": "gpt-test", "provider": "openai-codex"},
                                            "other": {"model": "other"}}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        key = store.get_or_create_session(source).session_key
        assert key.startswith("agent:main:")
        assert await bind_forum_post(runner, event) is None
        assert store.get_execution_profile(key) == "researcher"
        assert store.get_model_override(key) == {"model": "gpt-test", "provider": "openai-codex"}
        channel.applied_tags[:] = [SimpleNamespace(name="other")]
        assert await bind_forum_post(runner, event) is None
        assert store.get_model_override(key)["model"] == "gpt-test"
        store.close_all_db_handles()
        restarted = SessionStore(home / "sessions", GatewayConfig())
        assert restarted.get_execution_profile(key) == "researcher"
        assert restarted.get_model_override(key)["model"] == "gpt-test"
        restarted.close_all_db_handles()


@pytest.mark.asyncio
async def test_forum_rejects_ambiguous_tag_and_missing_profile_without_binding(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store), _session_model_overrides={})
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-2", chat_type="thread",
                           thread_id="thread-2", parent_chat_id="forum-1")
    channel = SimpleNamespace(applied_tags=[])
    event = MessageEvent(text="task", message_type=MessageType.TEXT, source=source,
                         raw_message=SimpleNamespace(channel=channel))
    config = {"gateway": {"spawn": {"parent_channel_id": "forum-1", "models": {
        "sol": {"model": "gpt-test", "profile": "researcher"}, "other": "other"}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        key = store.get_or_create_session(source).session_key
        assert "Select exactly one" in await bind_forum_post(runner, event)
        channel.applied_tags[:] = [SimpleNamespace(name="sol"), SimpleNamespace(name="other")]
        assert "Select exactly one" in await bind_forum_post(runner, event)
        channel.applied_tags[:] = [SimpleNamespace(name="sol")]
        assert "missing profile" in await bind_forum_post(runner, event)
        assert store.get_execution_profile(key) is None
        assert store.get_model_override(key) is None
    store.close_all_db_handles()
