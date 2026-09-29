"""A Discord workspace remains Main-owned while its execution/model survive a store reload."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.spawn_command import handle_spawn
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_spawn_binds_model_profile_before_first_task_and_retains_receiving_bot(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    satellite = home / "profiles" / "researcher"
    satellite.mkdir(parents=True)
    (satellite / "config.yaml").write_text("{}\n")
    (home / ".env").write_text("SPAWN_SCOPE_CHECK=main\n")
    (satellite / ".env").write_text("SPAWN_SCOPE_CHECK=researcher\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    adapter = SimpleNamespace(create_spawn_thread=AsyncMock(return_value={
        "success": True, "thread_id": "123", "thread_name": "News · sol", "guild_id": "guild"}),
        handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_model_overrides = {}
    runner._delivery_adapter_for = lambda source: adapter
    seen = []

    async def execute(*args, **kwargs):
        from agent.secret_scope import get_secret
        seen.append((get_hermes_home(), get_secret("SPAWN_SCOPE_CHECK")))
        return {"completed": True}

    runner._run_agent_inner = execute
    async def admit(event):
        event._gateway_accepted = True
    adapter.handle_message.side_effect = admit
    source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="channel",
                           user_id="owner", guild_id="guild")
    event = MessageEvent(text="/spawn --agent research --model sol --title 'News today' --prompt 'Check sources'",
                         source=source)
    config = {"gateway": {"spawn": {"parent_channel_id": "42", "agents": {"research": "researcher"},
                                 "models": {"sol": {"model": "gpt-test", "provider": "openai-codex"}}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        assert "Spawned <#123>" in await handle_spawn(runner, event)
    adapter.create_spawn_thread.assert_awaited_once()
    kickoff = adapter.handle_message.await_args.args[0]
    assert kickoff.text == "Check sources"
    assert kickoff.source.profile is None
    assert kickoff.source.chat_id == "123"
    key = store.get_or_create_session(kickoff.source).session_key
    assert key.startswith("agent:main:")
    assert store.get_model_override(key) == {"model": "gpt-test", "provider": "openai-codex"}
    assert store.get_execution_profile(key) == "researcher"
    assert (await runner._run_agent(kickoff.text, "", [], kickoff.source, "s1", session_key=key))["completed"]
    store.close_all_db_handles()
    reloaded = SessionStore(home / "sessions", GatewayConfig())
    assert reloaded.get_execution_profile(key) == "researcher"
    assert reloaded.get_model_override(key)["model"] == "gpt-test"
    runner.session_store = reloaded
    assert (await runner._run_agent("follow-up", "", [], kickoff.source, "s2", session_key=key))["completed"]
    assert seen == [(satellite, "researcher"), (satellite, "researcher")]
    reloaded.close_all_db_handles()


@pytest.mark.asyncio
async def test_spawn_refuses_missing_profile_and_invalid_alias_before_creation(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = SimpleNamespace(create_spawn_thread=AsyncMock())
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
    source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="channel")
    config = {"gateway": {"spawn": {"parent_channel_id": "42", "agents": {"research": "researcher"},
                                 "models": {"sol": "gpt-test"}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        assert "not installed" in await handle_spawn(runner, MessageEvent(text="/spawn research sol", source=source))
        assert "Unknown" in await handle_spawn(runner, MessageEvent(text="/spawn main nonexistent", source=source))
    adapter.create_spawn_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_forum_creation_requires_exact_model_tag_and_marks_owned_thread(monkeypatch):
    from plugins.platforms.discord import adapter as discord_module

    class Forum:
        def __init__(self):
            self.available_tags = [SimpleNamespace(name="sol")]
            self.create_thread = AsyncMock(return_value=SimpleNamespace(
                thread=SimpleNamespace(id=123, name="News · sol", guild=SimpleNamespace(id=456))))

    forum = Forum()
    monkeypatch.setattr(discord_module.discord, "ForumChannel", Forum)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="unused"))
    adapter._client = SimpleNamespace(get_channel=lambda _id: forum)
    adapter._threads = SimpleNamespace(mark_async=AsyncMock())
    for alias in ("other", None):
        refused = await adapter.create_spawn_thread("42", "News", "Starter", model_alias=alias)
        assert refused["success"] is False
        forum.create_thread.assert_not_awaited()
    created = await adapter.create_spawn_thread("42", "News", "Starter", model_alias="sol")
    assert created["thread_id"] == "123"
    assert forum.create_thread.await_args.kwargs["applied_tags"] == forum.available_tags
    adapter._threads.mark_async.assert_awaited_once_with("123")


@pytest.mark.asyncio
async def test_spawned_thread_enters_real_gateway_ingress_and_retains_scope_after_reload(tmp_path, monkeypatch):
    """A command-created thread's first task and a later inbound post use the same receiving lane."""
    home = tmp_path / ".hermes"
    satellite = home / "profiles" / "researcher"
    satellite.mkdir(parents=True)
    (satellite / "config.yaml").write_text("{}\n")
    (home / ".env").write_text("SPAWN_SCOPE_CHECK=main\n")
    (satellite / ".env").write_text("SPAWN_SCOPE_CHECK=researcher\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_model_overrides = {}
    runner._pending_messages = {}
    runner._background_tasks = set()
    runner._voice_mode = {}
    runner._draining = False
    runner._is_user_authorized = lambda _: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    receiving = SimpleNamespace(create_spawn_thread=AsyncMock(return_value={
        "success": True, "thread_id": "123", "thread_name": "News · sol", "guild_id": "guild"}))
    runner._delivery_adapter_for = lambda _: receiving
    seen = []

    async def model_turn(*args, **kwargs):
        from agent.secret_scope import get_secret
        seen.append((get_hermes_home(), get_secret("SPAWN_SCOPE_CHECK")))
        return "model reply"

    async def inbound_turn(self, event, source, key, generation):
        return await self._run_agent(event.text, "", [], source, "session", session_key=key)

    runner._run_agent_inner = model_turn
    async def admit(event):
        result = await runner._handle_message(event)
        event._gateway_accepted = result == "model reply"
    receiving.handle_message = admit
    source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="channel",
                           user_id="owner", guild_id="guild")
    command = MessageEvent(text="/spawn --agent research --model sol --prompt 'Check sources'", source=source)
    config = {"gateway": {"spawn": {"parent_channel_id": "42", "agents": {"research": "researcher"},
                                 "models": {"sol": {"model": "gpt-test", "provider": "openai-codex"}}}}}
    with (patch("gateway.run._load_gateway_config", return_value=config),
          patch.object(GatewayRunner, "_handle_message_with_agent", inbound_turn),
          patch.object(GatewayRunner, "_run_post_turn_hooks", AsyncMock()),
          patch.object(GatewayRunner, "_clear_durable_active_turn", AsyncMock()),
          patch.object(GatewayRunner, "_persist_active_agents", lambda self: None)):
        assert "Spawned <#123>" in await handle_spawn(runner, command)
        key = store.get_or_create_session(SessionSource(platform=Platform.DISCORD, chat_id="123",
            chat_type="thread", thread_id="123", parent_chat_id="42", guild_id="guild",
            scope_id="guild")).session_key
        assert store.get_execution_profile(key) == "researcher"
        store.close_all_db_handles()
        runner.session_store = SessionStore(home / "sessions", GatewayConfig())
        followup = MessageEvent(text="New question", source=SessionSource(
            platform=Platform.DISCORD, chat_id="123", chat_type="thread", thread_id="123",
            parent_chat_id="42", guild_id="guild", scope_id="guild", user_id="owner"))
        assert await runner._handle_message(followup) == "model reply"
    assert seen == [(satellite, "researcher"), (satellite, "researcher")]
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_spawn_reports_refused_initial_task_without_claiming_it_was_queued(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_model_overrides = {}
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="unused"))
    adapter.create_spawn_thread = AsyncMock(return_value={
        "success": True, "thread_id": "123", "guild_id": "guild"})
    # A connected adapter can lose its runner callback during shutdown: the real
    # admission path drops the kickoff even though the thread was created.
    runner._delivery_adapter_for = lambda _: adapter
    source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="channel",
                           user_id="owner", guild_id="guild")
    event = MessageEvent(text="/spawn --model sol --prompt 'Check sources'", source=source)
    config = {"gateway": {"spawn": {"parent_channel_id": "42",
                                  "models": {"sol": {"model": "gpt-test"}}}}}
    with patch("gateway.run._load_gateway_config", return_value=config):
        response = await handle_spawn(runner, event)
    assert "Spawned <#123>" in response
    assert "not admitted" in response
    assert "queued" not in response
    key = store.get_or_create_session(SessionSource(platform=Platform.DISCORD, chat_id="123",
        chat_type="thread", thread_id="123", parent_chat_id="42", guild_id="guild",
        scope_id="guild")).session_key
    assert store.get_model_override(key)["model"] == "gpt-test"
    assert store.get_execution_profile(key) == "default"
    store.close_all_db_handles()
