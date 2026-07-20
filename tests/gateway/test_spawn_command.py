"""Regression coverage for Main-brokered Discord /spawn workspaces."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, resolve_command


class SpawnHarness(GatewaySlashCommandsMixin):
    def __init__(self, adapter, store):
        self._adapter = adapter
        self.async_session_store = store
        self._session_model_overrides = {}
        self._background_tasks = set()
        self._handle_message = AsyncMock(return_value=None)

    def _adapter_for_source(self, _source):
        return self._adapter


def _discord_dm_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="dm-1",
            chat_type="dm",
            user_id="owner-1",
            user_name="Owner",
        ),
    )


def test_spawn_is_gateway_command_with_busy_session_handler():
    command = resolve_command("spawn")
    assert command is not None
    assert command.gateway_only is True
    assert "spawn" in ACTIVE_SESSION_BYPASS_COMMANDS


def test_session_entry_roundtrip_persists_execution_profile():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
    )
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        execution_profile="Researcher",
        model_override={
            "model": "glm-5.2",
            "provider": "zai",
            "api_key": "must-not-persist",
        },
    )

    payload = entry.to_dict()
    assert payload["execution_profile"] == "researcher"
    assert payload["model_override"] == {"model": "glm-5.2", "provider": "zai"}

    restored = SessionEntry.from_dict(payload)
    assert restored.execution_profile == "researcher"
    assert restored.model_override == {"model": "glm-5.2", "provider": "zai"}


def test_session_reset_preserves_spawn_binding(tmp_path):
    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = None
    store._loaded = True

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
    )
    key = build_session_key(source)
    store._entries[key] = SessionEntry(
        session_key=key,
        session_id="old-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        execution_profile="researcher",
        model_override={"model": "glm-5.2", "provider": "zai"},
    )

    reset = store.reset_session(key)

    assert reset is not None
    assert reset.execution_profile == "researcher"
    assert reset.model_override == {"model": "glm-5.2", "provider": "zai"}
    assert store.get_execution_profile(key) == "researcher"


@pytest.mark.asyncio
async def test_spawn_creates_thread_and_persists_profile_and_model_alias():
    adapter = SimpleNamespace(
        create_spawn_thread=AsyncMock(
            return_value={
                "success": True,
                "thread_id": "thread-9",
                "thread_name": "Research task",
                "guild_id": "guild-1",
            }
        )
    )
    spawned_entry = SimpleNamespace(
        session_key="agent:main:discord:thread:thread-9:thread-9"
    )
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=spawned_entry),
        set_execution_profile=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "parent-1",
                "agents": {"research": {"profile": "researcher"}},
                "models": {
                    "glm52": {"model": "glm-5.2", "provider": "zai"}
                },
            }
        }
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event('/spawn research glm52 "Research task"')
        )

    assert result.startswith("✅ Spawned <#thread-9>")
    adapter.create_spawn_thread.assert_awaited_once()
    create_kwargs = adapter.create_spawn_thread.await_args.kwargs
    assert create_kwargs["parent_chat_id"] == "parent-1"
    assert create_kwargs["name"] == "Research task"
    assert create_kwargs["owner_user_id"] == "owner-1"

    spawned_source = store.get_or_create_session.await_args.args[0]
    assert spawned_source.chat_id == "thread-9"
    assert spawned_source.thread_id == "thread-9"
    assert spawned_source.parent_chat_id == "parent-1"
    assert spawned_source.profile is None
    assert spawned_source.scope_id == "guild-1"
    store.set_execution_profile.assert_awaited_once_with(
        spawned_entry.session_key, "researcher"
    )
    store.set_model_override.assert_awaited_once_with(
        spawned_entry.session_key,
        {"model": "glm-5.2", "provider": "zai"},
    )
    runner._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_with_prompt_posts_and_dispatches_initial_task():
    adapter = SimpleNamespace(
        create_spawn_thread=AsyncMock(
            return_value={
                "success": True,
                "thread_id": "thread-10",
                "thread_name": "Investigate build",
                "guild_id": "guild-1",
            }
        ),
        send=AsyncMock(return_value=SimpleNamespace(success=True)),
        handle_message=AsyncMock(),
    )
    spawned_entry = SimpleNamespace(
        session_key="agent:main:discord:thread:thread-10:thread-10"
    )
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=spawned_entry),
        set_execution_profile=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "parent-1",
                "agents": {"research": {"profile": "researcher"}},
                "models": {
                    "glm52": {"model": "glm-5.2", "provider": "zai"}
                },
            }
        }
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event(
                "/spawn --agent research --model glm52 "
                "--title 'Investigate build' --prompt 'Inspect the failing CI job'"
            )
        )

    assert result.startswith("✅ Spawned <#thread-10>")
    assert "Initial task: queued" in result
    adapter.send.assert_awaited_once()
    send_args = adapter.send.await_args
    assert send_args is not None
    assert send_args.args[0] == "thread-10"
    assert "Inspect the failing CI job" in send_args.args[1]
    assert send_args.kwargs["metadata"]["thread_id"] == "thread-10"

    adapter.handle_message.assert_awaited_once()
    kickoff = adapter.handle_message.await_args.args[0]
    assert kickoff.text == "Inspect the failing CI job"
    assert kickoff.message_type == MessageType.TEXT
    assert kickoff.source.chat_id == "thread-10"
    assert kickoff.source.thread_id == "thread-10"
    assert kickoff.source.profile is None
    runner._handle_message.assert_not_awaited()
    store.set_execution_profile.assert_awaited_once_with(
        spawned_entry.session_key, "researcher"
    )
    store.set_model_override.assert_awaited_once_with(
        spawned_entry.session_key,
        {"model": "glm-5.2", "provider": "zai"},
    )


@pytest.mark.asyncio
async def test_create_spawn_thread_adds_owner_without_member_cache():
    from plugins.platforms.discord.adapter import DiscordAdapter

    thread = SimpleNamespace(
        id=12345,
        name="Visible task",
        guild=SimpleNamespace(id=67890),
        send=AsyncMock(),
        add_user=AsyncMock(),
    )
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.create_handoff_thread = AsyncMock(return_value="12345")
    adapter._client = SimpleNamespace(get_channel=lambda _channel_id: thread)
    object.__setattr__(
        adapter, "_threads", SimpleNamespace(mark=lambda _thread_id: None)
    )
    spawn_owner_mark = MagicMock()
    object.__setattr__(
        adapter, "_spawn_owners", SimpleNamespace(mark=spawn_owner_mark)
    )

    with patch(
        "plugins.platforms.discord.adapter.discord.Object",
        side_effect=lambda *, id: SimpleNamespace(id=id),
    ):
        result = await DiscordAdapter.create_spawn_thread(
            adapter,
            parent_chat_id="999",
            name="Visible task",
            starter_content="Workspace ready",
            owner_user_id="210318156432932864",
        )

    assert result["success"] is True
    thread.add_user.assert_awaited_once()
    member = thread.add_user.await_args.args[0]
    assert str(member.id) == "210318156432932864"
    spawn_owner_mark.assert_called_once_with("12345:210318156432932864")


@pytest.mark.asyncio
async def test_spawn_rejects_uninstalled_agent_without_creating_thread():
    adapter = SimpleNamespace(create_spawn_thread=AsyncMock())
    store = SimpleNamespace()
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {"spawn": {"parent_channel_id": "parent-1"}}
    }
    profiles = [SimpleNamespace(name="default"), SimpleNamespace(name="researcher")]

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=False),
        patch("hermes_cli.profiles.list_profiles", return_value=profiles),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event("/spawn heman kimi3")
        )

    assert "profile `heman`" in result
    assert "not installed" in result
    adapter.create_spawn_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_rejects_unconfigured_model_alias_without_side_effects():
    adapter = SimpleNamespace(create_spawn_thread=AsyncMock())
    runner = SpawnHarness(adapter, SimpleNamespace())
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "parent-1",
                "models": {"kimi3": {"provider": "opencode-go", "model": "kimi-k3"}},
            }
        }
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event("/spawn main arbitrary-provider/model")
        )

    assert "Unknown model alias" in result
    assert "`kimi3`" in result
    adapter.create_spawn_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_scopes_spawn_profile_but_keeps_main_source(tmp_path):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    store = SimpleNamespace()
    execution_profiles = AsyncMock(return_value="researcher")
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_execution_profile=execution_profiles,
    )
    runner._run_agent_inner = AsyncMock(return_value={"final_response": "ok"})
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        profile=None,
    )
    entered = []

    @contextmanager
    def fake_scope(home):
        entered.append(Path(home))
        yield

    profile_home = tmp_path / "profiles" / "researcher"
    with (
        patch("hermes_cli.profiles.profile_exists", return_value=True),
        patch("hermes_cli.profiles.get_profile_dir", return_value=profile_home),
        patch("gateway.run._profile_runtime_scope", fake_scope),
    ):
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:discord:thread:thread-1:thread-1",
        )

    assert result == {"final_response": "ok"}
    execution_profiles.assert_awaited_once_with(
        "agent:main:discord:thread:thread-1:thread-1"
    )
    assert entered == [profile_home]
    assert source.profile is None
    runner._run_agent_inner.assert_awaited_once()
