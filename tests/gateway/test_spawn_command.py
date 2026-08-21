"""Regression coverage for Main-brokered Discord /spawn workspaces."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.base import utf16_len
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from gateway.slash_commands import (
    GatewaySlashCommandsMixin,
    _spawn_model_alias,
    _spawn_topic_essence,
    _spawn_topic_title,
)
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

    def _session_key_for_source(self, source):
        return build_session_key(source)


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


def test_spawn_topic_title_limits_essence_and_uses_friendly_model():
    request = "Please analyze the current Discord gateway reliability settings"
    assert _spawn_topic_essence(request) == "analyze Discord gateway reliability"
    assert _spawn_topic_title(request, "Gemini 3.7 Flash High") == (
        "analyze Discord gateway reliability · Gemini 3.7 Flash High"
    )
    long_title = _spawn_topic_title("😀 reliability audit settings", "X" * 96)
    assert utf16_len(long_title) <= 80
    assert " · " in long_title


def test_spawn_model_alias_does_not_guess_shared_wire_model():
    config = {
        "gateway": {
            "spawn": {
                "models": {
                    "sol-high": {"model": "gpt-5.6-sol"},
                    "sol-xhigh": {"model": "gpt-5.6-sol"},
                }
            }
        }
    }
    assert _spawn_model_alias(config, "sol-xhigh", "gpt-5.6-sol") == "sol-xhigh"
    assert _spawn_model_alias(config, "gpt-5.6-sol", "gpt-5.6-sol") == ""


@pytest.mark.asyncio
async def test_bound_forum_rejects_persistent_model_without_unique_tag_alias():
    store = SimpleNamespace(get_execution_profile=AsyncMock(return_value="default"))
    runner = SpawnHarness(SimpleNamespace(), store)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
    )
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {
                    "sol-high": {"model": "gpt-5.6-sol"},
                    "sol-xhigh": {"model": "gpt-5.6-sol"},
                },
            }
        }
    }

    error = await runner._forum_topic_model_alias_error(
        source,
        raw_config=config,
        requested_model="gpt-5.6-sol",
        resolved_model="gpt-5.6-sol",
    )
    assert "no unique forum tag" in error

    exact = await runner._forum_topic_model_alias_error(
        source,
        raw_config=config,
        requested_model="sol-xhigh",
        resolved_model="gpt-5.6-sol",
    )
    assert exact == ""


@pytest.mark.asyncio
async def test_spawn_topic_model_sync_preserves_essence_and_renames():
    adapter = SimpleNamespace(
        rename_thread=AsyncMock(return_value={"success": True}),
        set_forum_thread_model_tag=AsyncMock(return_value=True),
    )
    store = SimpleNamespace(get_execution_profile=AsyncMock(return_value="default"))
    runner = SpawnHarness(adapter, store)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name="Hermes config audit · Sol",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
    )
    config = {
        "gateway": {"spawn": {"parent_channel_id": "forum-1"}},
        "model_aliases": {
            "sol-high": {
                "label": "Sol",
                "model": "gpt-5.6-sol",
            },
            "gemini": {
                "label": "Gemini 3.7 Flash High",
                "model": "gemini-3.7-flash-high",
            }
        },
    }

    await runner._sync_spawn_topic_model_title(
        source,
        raw_config=config,
        requested_model="gemini",
        resolved_model="gemini-3.7-flash-high",
    )

    adapter.rename_thread.assert_awaited_once_with(
        "thread-1",
        "Hermes config audit · Gemini 3.7 Flash High",
        only_if_current_name="Hermes config audit · Sol",
    )
    adapter.set_forum_thread_model_tag.assert_awaited_once_with(
        "thread-1",
        "gemini",
        model_aliases={"sol-high", "gemini"},
    )


@pytest.mark.asyncio
async def test_spawn_topic_model_sync_reports_tag_failure():
    adapter = SimpleNamespace(
        rename_thread=AsyncMock(return_value=True),
        set_forum_thread_model_tag=AsyncMock(return_value=False),
    )
    store = SimpleNamespace(get_execution_profile=AsyncMock(return_value="default"))
    runner = SpawnHarness(adapter, store)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name="Audit · Sol",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
    )
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {"sol": {"label": "Sol", "model": "sol-wire"}},
            }
        },
        "model_aliases": {
            "gemini": {"label": "Gemini", "model": "gemini-wire"}
        },
    }

    warning = await runner._sync_spawn_topic_model_title(
        source,
        raw_config=config,
        requested_model="gemini",
        resolved_model="gemini-wire",
    )

    assert "model tag" in warning
    adapter.set_forum_thread_model_tag.assert_awaited_once_with(
        "thread-1",
        "gemini",
        model_aliases={"sol", "gemini"},
    )


@pytest.mark.asyncio
async def test_forum_model_tag_binds_before_first_turn():
    channel = SimpleNamespace(
        name="Audit my Hermes configuration",
        applied_tags=[SimpleNamespace(name="sol-high")],
    )
    raw_message = SimpleNamespace(channel=channel)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name="Guild / 🌐chats / Audit my Hermes configuration",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
        user_id="owner-1",
    )
    event = MessageEvent(
        text="Analyze the current Hermes configuration.",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw_message,
    )
    entry = SimpleNamespace(session_key=build_session_key(source))
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=entry),
        get_execution_profile=AsyncMock(return_value=None),
        set_execution_profile=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    adapter = SimpleNamespace(
        rename_thread=AsyncMock(return_value=True),
        _threads=SimpleNamespace(mark=MagicMock()),
        _spawn_owners=SimpleNamespace(mark=MagicMock()),
    )
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "default_agent": "main",
                "models": {
                    "sol-high": {
                        "label": "Sol High",
                        "model": "gpt-5.6-sol",
                        "provider": "openai-codex",
                        "reasoning_effort": "high",
                    }
                },
            }
        }
    }

    with patch("gateway.run._load_gateway_config", return_value=config), patch(
        "hermes_cli.profiles.profile_exists", return_value=True
    ):
        result = await runner._prepare_discord_forum_tag_session(event)

    assert result is None
    store.set_execution_profile.assert_awaited_once_with(entry.session_key, "default")
    store.set_model_override.assert_awaited_once_with(
        entry.session_key,
        {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "reasoning_effort": "high",
        },
    )
    adapter._threads.mark.assert_called_once_with("thread-1")
    adapter._spawn_owners.mark.assert_called_once_with("thread-1:owner-1")
    adapter.rename_thread.assert_awaited_once_with(
        "thread-1",
        "Audit Hermes configuration · Sol High",
        only_if_current_name="Audit my Hermes configuration",
    )


@pytest.mark.asyncio
async def test_forum_placeholder_waits_for_semantic_title_before_rename():
    channel = SimpleNamespace(
        name=".",
        applied_tags=[SimpleNamespace(name="sol-high")],
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
        user_id="owner-1",
    )
    event = MessageEvent(
        text="Fix dot title inference.",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=SimpleNamespace(channel=channel),
    )
    entry = SimpleNamespace(session_key=build_session_key(source))
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=entry),
        get_execution_profile=AsyncMock(return_value=None),
        set_execution_profile=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    adapter = SimpleNamespace(
        rename_thread=AsyncMock(return_value=True),
        _threads=SimpleNamespace(mark=MagicMock()),
        _spawn_owners=SimpleNamespace(mark=MagicMock()),
    )
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "default_agent": "main",
                "models": {
                    "sol-high": {
                        "label": "Sol High",
                        "model": "gpt-5.6-sol",
                        "provider": "openai-codex",
                        "reasoning_effort": "high",
                    }
                },
            }
        }
    }

    with patch("gateway.run._load_gateway_config", return_value=config), patch(
        "hermes_cli.profiles.profile_exists", return_value=True
    ):
        result = await runner._prepare_discord_forum_tag_session(event)

    assert result is None
    store.set_execution_profile.assert_awaited_once_with(entry.session_key, "default")
    store.set_model_override.assert_awaited_once()
    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_forum_semantic_title_performs_the_single_guarded_rename():
    thread = SimpleNamespace(name=".")
    adapter = SimpleNamespace(
        _client=SimpleNamespace(fetch_channel=AsyncMock(return_value=thread)),
        rename_thread=AsyncMock(return_value=True),
    )

    class RenameHarness:
        _is_discord_spawn_forum_lane = GatewayRunner._is_discord_spawn_forum_lane
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _is_relay_discord_channel_lane = GatewayRunner._is_relay_discord_channel_lane
        _await_relay_auto_thread_info = GatewayRunner._await_relay_auto_thread_info
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )

        def __init__(self):
            self.adapters = {Platform.DISCORD: adapter}
            self.async_session_store = SimpleNamespace(
                get_model_override=AsyncMock(
                    return_value={"model": "gpt-5.6-sol"}
                )
            )

        def _adapter_for_source(self, _source):
            return adapter

        def _session_key_for_source(self, source):
            return build_session_key(source)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="12345",
        chat_type="thread",
        thread_id="12345",
        parent_chat_id="forum-1",
        user_id="owner-1",
    )
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {
                    "sol-high": {
                        "label": "Sol High",
                        "model": "gpt-5.6-sol",
                    }
                },
            }
        }
    }

    with patch("gateway.run._load_gateway_config", return_value=config):
        await RenameHarness()._rename_discord_auto_thread_for_session_title(
            source,
            "session-1",
            "Fix dot title inference",
        )

    adapter.rename_thread.assert_awaited_once_with(
        "12345",
        "Fix dot title inference · Sol High",
        prefer_connector_created=False,
        only_if_current_name=".",
        parent_chat_id=None,
    )


@pytest.mark.asyncio
async def test_forum_model_tag_requires_exactly_one_model():
    channel = SimpleNamespace(
        name="Audit",
        applied_tags=[
            SimpleNamespace(name="sol-high"),
            SimpleNamespace(name="gemini"),
        ],
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
        user_id="owner-1",
    )
    event = MessageEvent(
        text="Audit",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=SimpleNamespace(channel=channel),
    )
    entry = SimpleNamespace(session_key=build_session_key(source))
    store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=entry),
        get_execution_profile=AsyncMock(return_value=None),
        set_execution_profile=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    runner = SpawnHarness(SimpleNamespace(), store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {
                    "sol-high": {"model": "sol"},
                    "gemini": {"model": "gemini"},
                },
            }
        }
    }

    with patch("gateway.run._load_gateway_config", return_value=config):
        result = await runner._prepare_discord_forum_tag_session(event)

    assert result is not None and "exactly one model tag" in result
    store.set_execution_profile.assert_not_awaited()
    store.set_model_override.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution_profile", "chat_name"),
    [
        (None, "Manual forum post · Sol"),
        ("default", "Human custom title"),
        ("default", "Human custom title · Personal"),
    ],
)
async def test_spawn_topic_model_sync_does_not_clobber_manual_titles(
    execution_profile, chat_name
):
    adapter = SimpleNamespace(rename_thread=AsyncMock(return_value=True))
    store = SimpleNamespace(
        get_execution_profile=AsyncMock(return_value=execution_profile)
    )
    runner = SpawnHarness(adapter, store)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name=chat_name,
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
    )
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {"sol-high": {"label": "Sol High", "model": "sol"}},
            }
        },
        "model_aliases": {"sol-high": {"label": "Sol", "model": "sol"}},
    }

    await runner._sync_spawn_topic_model_title(
        source,
        raw_config=config,
        requested_model="sol-high",
        resolved_model="sol",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_topic_model_sync_recognizes_truncated_long_model_suffix():
    old_label = "O" * 96
    current_name = _spawn_topic_title("Audit", old_label)
    adapter = SimpleNamespace(rename_thread=AsyncMock(return_value=True))
    store = SimpleNamespace(get_execution_profile=AsyncMock(return_value="default"))
    runner = SpawnHarness(adapter, store)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name=current_name,
        chat_type="thread",
        thread_id="thread-1",
        parent_chat_id="forum-1",
    )
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "forum-1",
                "models": {
                    "old": {"label": old_label, "model": "old-wire"},
                    "new": {"label": "New Model", "model": "new-wire"},
                },
            }
        }
    }

    await runner._sync_spawn_topic_model_title(
        source,
        raw_config=config,
        requested_model="new",
        resolved_model="new-wire",
    )

    adapter.rename_thread.assert_awaited_once_with(
        "thread-1",
        "Au · New Model",
        only_if_current_name=current_name,
    )


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
                    "glm52": {
                        "model": "glm-5.2",
                        "provider": "zai",
                        "reasoning_effort": "high",
                    }
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
    assert create_kwargs["name"] == "Research task · glm52"
    assert create_kwargs["owner_user_id"] == "owner-1"
    assert create_kwargs["model_alias"] == "glm52"

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
        {"model": "glm-5.2", "provider": "zai", "reasoning_effort": "high"},
    )
    runner._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawn_uses_configured_default_model_and_specific_label():
    adapter = SimpleNamespace(
        create_spawn_thread=AsyncMock(
            return_value={
                "success": True,
                "thread_id": "thread-default",
                "thread_name": "Default choice · Sol High",
                "guild_id": "guild-1",
            }
        )
    )
    spawned_entry = SimpleNamespace(session_key="spawn-default-key")
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
                "default_model": "sol-high",
                "models": {
                    "sol-high": {
                        "label": "Sol High",
                        "model": "gpt-5.6-sol",
                        "provider": "openai-codex",
                        "reasoning_effort": "high",
                    }
                },
            }
        },
        "model_aliases": {
            "sol-high": {
                "label": "Sol",
                "model": "gpt-5.6-sol",
                "provider": "openai-codex",
                "reasoning_effort": "high",
            }
        },
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event('/spawn --title "Default choice"')
        )

    assert "Model: `Sol High`" in result
    create_kwargs = adapter.create_spawn_thread.await_args.kwargs
    assert create_kwargs["name"] == "Default choice · Sol High"
    store.set_model_override.assert_awaited_once_with(
        spawned_entry.session_key,
        {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "reasoning_effort": "high",
        },
    )


@pytest.mark.asyncio
async def test_global_model_alias_can_bind_dedicated_execution_profile():
    adapter = SimpleNamespace(
        create_spawn_thread=AsyncMock(
            return_value={
                "success": True,
                "thread_id": "thread-gemini",
                "thread_name": "Gemini task",
                "guild_id": "guild-1",
            }
        )
    )
    spawned_entry = SimpleNamespace(session_key="agent:main:discord:thread:gemini")
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
                "default_agent": "main",
                "agents": {"main": {"profile": "default"}},
                "models": {
                    "gemini": {
                        "profile": "gemini",
                        "model": "gemini-3.6-flash-high",
                        "provider": "copilot-acp",
                        "base_url": "acp://antigravity",
                    }
                },
            }
        }
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event('/spawn --model gemini --title "Gemini task"')
        )

    assert result.startswith("✅ Spawned <#thread-gemini>")
    create_kwargs = adapter.create_spawn_thread.await_args.kwargs
    assert create_kwargs["name"] == "Gemini task · gemini"
    store.set_execution_profile.assert_awaited_once_with(
        spawned_entry.session_key, "gemini"
    )
    store.set_model_override.assert_awaited_once_with(
        spawned_entry.session_key,
        {
            "model": "gemini-3.6-flash-high",
            "provider": "copilot-acp",
            "base_url": "acp://antigravity",
        },
    )


@pytest.mark.asyncio
async def test_model_alias_missing_profile_fails_before_thread_creation():
    adapter = SimpleNamespace(create_spawn_thread=AsyncMock())
    store = SimpleNamespace()
    runner = SpawnHarness(adapter, store)
    config = {
        "gateway": {
            "spawn": {
                "parent_channel_id": "parent-1",
                "models": {
                    "gemini": {
                        "profile": "missing-gemini",
                        "model": "gemini-3.6-flash-high",
                    }
                },
            }
        }
    }

    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch(
            "hermes_cli.profiles.profile_exists",
            side_effect=lambda name: name == "default",
        ),
    ):
        result = await runner._handle_spawn_command(
            _discord_dm_event("/spawn --model gemini")
        )

    assert "profile `missing-gemini`" in result
    assert "not installed" in result
    adapter.create_spawn_thread.assert_not_awaited()


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
                    "glm52": {
                        "model": "glm-5.2",
                        "provider": "zai",
                        "reasoning_effort": "high",
                    }
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
        {"model": "glm-5.2", "provider": "zai", "reasoning_effort": "high"},
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
async def test_create_spawn_thread_uses_selected_tag_in_required_tag_forum():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class FakeForumChannel:
        def __init__(self):
            self.available_tags = [SimpleNamespace(id=7, name="sol-high")]
            self.create_thread = AsyncMock()

    thread = SimpleNamespace(
        id=12345,
        name="Visible task · Sol High",
        guild=SimpleNamespace(id=67890),
        send=AsyncMock(),
        add_user=AsyncMock(),
    )
    parent = FakeForumChannel()
    parent.create_thread.return_value = SimpleNamespace(thread=thread)
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: parent if channel_id == 999 else thread,
        fetch_channel=AsyncMock(),
    )
    object.__setattr__(adapter, "_threads", SimpleNamespace(mark=MagicMock()))
    object.__setattr__(adapter, "_spawn_owners", SimpleNamespace(mark=MagicMock()))

    with patch(
        "plugins.platforms.discord.adapter.discord.ForumChannel", FakeForumChannel
    ):
        result = await DiscordAdapter.create_spawn_thread(
            adapter,
            parent_chat_id="999",
            name="Visible task · Sol High",
            starter_content="Workspace ready",
            model_alias="sol-high",
        )

    assert result["success"] is True
    parent.create_thread.assert_awaited_once_with(
        name="Visible task · Sol High",
        content="Workspace ready",
        applied_tags=[parent.available_tags[0]],
        auto_archive_duration=1440,
        reason="Hermes /spawn workspace",
    )
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_spawn_thread_updates_discord_name():
    from plugins.platforms.discord.adapter import DiscordAdapter

    thread = SimpleNamespace(id=12345, name="Old · Sol", edit=AsyncMock())
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: thread,
        fetch_channel=AsyncMock(return_value=thread),
    )

    result = await DiscordAdapter.rename_thread(
        adapter, "12345", "Audit configs · Gemini 3.7 Flash High"
    )

    assert result is True
    thread.edit.assert_awaited_once_with(
        name="Audit configs · Gemini 3.7 Flash High",
        reason="Hermes semantic session title",
    )


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
