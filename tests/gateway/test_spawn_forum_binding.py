"""Native forum creation binds brokered credentials before the first model turn."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.spawn_forum import bind_forum_post
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


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
        store.set_model_override(key, None)  # /model resets to profile default
        assert await bind_forum_post(runner, event) is None
        assert store.get_model_override(key) is None
        assert store.get_execution_profile(key) == "researcher"
        store.close_all_db_handles()
        restarted = SessionStore(home / "sessions", GatewayConfig())
        assert restarted.get_execution_profile(key) == "researcher"
        assert restarted.get_model_override(key) is None
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


@pytest.mark.asyncio
async def test_forum_uses_agent_model_catalog_and_refuses_casefold_collisions(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    satellite = home / "profiles" / "researcher"
    satellite.mkdir(parents=True)
    (satellite / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store), _session_model_overrides={})
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-agent", chat_type="thread",
                           thread_id="thread-agent", parent_chat_id="forum-1")
    channel = SimpleNamespace(applied_tags=[SimpleNamespace(name="SOL")])
    event = MessageEvent(text="task", message_type=MessageType.TEXT, source=source,
                         raw_message=SimpleNamespace(channel=channel))
    config = {"gateway": {"spawn": {"parent_channel_id": "forum-1", "default_agent": "research",
                                 "agents": {"research": {"profile": "researcher", "models": {
                                     "sol": {"model": "agent-model", "provider": "openai-codex"},
                                     "private": {"model": "agent-only"}}}},
                                 "models": {"sol": {"model": "global-model"}}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        key = store.get_or_create_session(source).session_key
        assert await bind_forum_post(runner, event) is None
        assert store.get_model_override(key) == {"model": "agent-model", "provider": "openai-codex"}
        assert store.get_execution_profile(key) == "researcher"
        store.set_execution_profile(key, None)
        store.set_model_override(key, None)
        channel.applied_tags[:] = [SimpleNamespace(name="private")]
        assert await bind_forum_post(runner, event) is None
        assert store.get_model_override(key) == {"model": "agent-only"}
        store.set_execution_profile(key, None)
        store.set_model_override(key, None)
        config["gateway"]["spawn"]["models"]["SOL"] = "colliding-model"
        channel.applied_tags[:] = [SimpleNamespace(name="sol")]
        error = await bind_forum_post(runner, event)
        assert error is not None and "Select exactly one" in error
        assert store.get_execution_profile(key) is None
        assert store.get_model_override(key) is None
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_native_forum_first_post_and_followups_use_bound_secrets_after_reload(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    satellite = home / "profiles" / "researcher"
    satellite.mkdir(parents=True)
    (satellite / "config.yaml").write_text("{}\n")
    (home / ".env").write_text("SPAWN_SCOPE_CHECK=main\n")
    (satellite / ".env").write_text("SPAWN_SCOPE_CHECK=researcher\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = {"gateway": {"spawn": {"parent_channel_id": "forum-1",
                                 "default_agent": "research", "agents": {"research": "researcher"},
                                 "models": {"sol": {"model": "gpt-test"}}}}}
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-1", chat_type="thread",
                           thread_id="thread-1", parent_chat_id="forum-1")
    channel = SimpleNamespace(applied_tags=[SimpleNamespace(name="sol")])
    event = MessageEvent(text="first task", message_type=MessageType.TEXT, source=source,
                         raw_message=SimpleNamespace(channel=channel))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._session_model_overrides = {}
    runner.session_store = SessionStore(home / "sessions", GatewayConfig())
    seen = []

    async def execute(*args, **kwargs):
        from agent.secret_scope import get_secret
        seen.append((get_hermes_home(), get_secret("SPAWN_SCOPE_CHECK"), source.profile))
        return {"completed": True}

    runner._run_agent_inner = execute
    with patch("gateway.run._load_gateway_config", return_value=config):
        assert await bind_forum_post(runner, event) is None
        key = runner.session_store.get_or_create_session(source).session_key
        assert key.startswith("agent:main:")
        assert (await runner._run_agent("first task", "", [], source, "s1", session_key=key))["completed"]
        runner.session_store.close_all_db_handles()
        runner.session_store = SessionStore(home / "sessions", GatewayConfig())
        channel.applied_tags[:] = [SimpleNamespace(name="unknown")]
        assert await bind_forum_post(runner, event) is None
        assert (await runner._run_agent("follow-up", "", [], source, "s2", session_key=key))["completed"]
        assert runner.session_store.get_model_override(key) == {"model": "gpt-test"}
        runner.session_store.set_execution_profile(key, None)
        assert (await runner._run_agent("unbound", "", [], source, "s3", session_key=key))["completed"]
    assert seen == [(satellite, "researcher", None), (satellite, "researcher", None),
                    (home, None, None)]
    runner.session_store.close_all_db_handles()
