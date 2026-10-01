"""Native forum creation binds brokered credentials before the first model turn."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.spawn_forum import bind_forum_post, sync_switched_model_tag
from gateway.run import GatewayRunner
from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext
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
        channel.applied_tags[:] = [SimpleNamespace(name="sol"), SimpleNamespace(name="SOL")]
        duplicate_error = await bind_forum_post(runner, event)
        assert duplicate_error is not None and "Select exactly one" in duplicate_error
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


@pytest.mark.asyncio
async def test_bound_forum_model_switch_syncs_receiving_bot_without_changing_execution(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "researcher").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread",
                           thread_id="thread", parent_chat_id="forum")
    key = store.get_or_create_session(source).session_key
    store.set_execution_profile(key, "researcher")
    store.set_model_override(key, {"model": "old"})
    receiving = SimpleNamespace(set_forum_thread_model_tag=AsyncMock(return_value=True))
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store),
                             _delivery_adapter_for=lambda _: receiving)
    config = {"gateway": {"spawn": {"parent_channel_id": "forum", "default_agent": "research",
                                 "agents": {"research": {"profile": "researcher", "models": {
                                     "new": {"model": "target", "provider": "openai-codex"}}}},
                                 "models": {"old": "old", "wrong-scope": {
                                     "model": "target", "profile": "gemini"}}}}}
    result = SimpleNamespace(new_model="target", target_provider="openai-codex", base_url="")
    with patch("gateway.run._load_gateway_config", return_value=config):
        assert await sync_switched_model_tag(runner, source, key, result) is None
        receiving.set_forum_thread_model_tag.assert_awaited_once_with(
            "thread", "new", model_aliases={"new", "old", "wrong-scope"},
            title_label="new", known_labels={"new", "old", "wrong-scope"})
        assert store.get_execution_profile(key) == "researcher"
        assert store.get_model_override(key) == {"model": "old"}
        config["gateway"]["spawn"]["models"]["duplicate"] = "target"
        warning = await sync_switched_model_tag(runner, source, key, result)
        assert warning is not None and "no unique alias" in warning
        receiving.set_forum_thread_model_tag.assert_awaited_once()
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_model_commit_warns_if_forum_metadata_fails_after_persistence(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread",
                           thread_id="thread", parent_chat_id="forum")
    key = store.get_or_create_session(source).session_key
    store.set_execution_profile(key, "default")
    receiving = SimpleNamespace(set_forum_thread_model_tag=AsyncMock(return_value=False))

    class CommitRunner(GatewayModelCommandsMixin):
        async_session_store = AsyncSessionStore(store)

        def _switch_cached_agent_model(self, result, ctx, picker):
            return None

        def _record_switch_metrics(self, result, ctx, source):
            return None

        async def _record_model_switch(self, result, ctx, **kwargs):
            await self.async_session_store.set_model_override(ctx.session_key, {"model": result.new_model})
            return None

        async def _model_switch_confirmation(self, *args, **kwargs):
            return "switched"

        def _delivery_adapter_for(self, _):
            return receiving

    runner = CommitRunner()
    ctx = _ModelSwitchContext(session_key=key, source=source, config_path=home / "config.yaml",
                              persist_global=False)
    config = {"gateway": {"spawn": {"parent_channel_id": "forum", "models": {"new": "target"}}}}
    result = SimpleNamespace(new_model="target", target_provider="openai-codex", base_url="")
    with patch("gateway.run._load_gateway_config", return_value=config):
        reply = await runner._commit_model_switch_locked(result, ctx, source=source, picker=False)
    assert "switched" in reply and "Forum tag unchanged" in reply
    assert store.get_model_override(key) == {"model": "target"}
    assert store.get_execution_profile(key) == "default"
    store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("one_turn", [True, False])
async def test_non_session_model_switch_keeps_forum_tag_and_brokered_scope(tmp_path, monkeypatch, one_turn):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread",
                           thread_id="thread", parent_chat_id="forum")
    key = store.get_or_create_session(source).session_key
    store.set_execution_profile(key, "default")
    store.set_model_override(key, {"model": "standing"})
    receiving = SimpleNamespace(set_forum_thread_model_tag=AsyncMock())

    class Runner(GatewayModelCommandsMixin):
        async_session_store = AsyncSessionStore(store)
        _session_model_overrides = {}

        def _switch_cached_agent_model(self, result, ctx, picker):
            return None

        def _record_switch_metrics(self, result, ctx, source):
            return None

        def _evict_cached_agent(self, session_key):
            return None

        def _claim_one_turn_restore(self, session_key, snapshot):
            self.restore = snapshot

        def _channel_override_for(self, source):
            return None

        async def _model_switch_confirmation(self, *args, **kwargs):
            return "switched"

        def _delivery_adapter_for(self, source):
            return receiving

    runner = Runner()
    ctx = _ModelSwitchContext(session_key=key, source=source, config_path=home / "config.yaml",
                              persist_global=not one_turn, one_turn=one_turn,
                              restore_snapshot={"model": "standing"}, current_model="standing")
    result = SimpleNamespace(new_model="temporary" if one_turn else "global", target_provider="openai-codex",
                             provider_label="Codex", api_key="", base_url="", api_mode="chat_completions",
                             request_overrides={}, runtime_capabilities={})
    with patch("gateway.run._load_gateway_config", return_value={
        "gateway": {"spawn": {"parent_channel_id": "forum", "models": {"standing": "standing"}}},
    }), patch("gateway.slash_commands_model._persist_model_switch_to_config", new_callable=AsyncMock) as persist:
        reply = await runner._commit_model_switch_locked(result, ctx, source=source, picker=False)
    assert "Forum tag unchanged" in reply
    assert ("next turn only" if one_turn else "global configuration") in reply
    receiving.set_forum_thread_model_tag.assert_not_awaited()
    assert store.get_execution_profile(key) == "default"
    assert store.get_model_override(key) == ({"model": "standing"} if one_turn else None)
    if one_turn:
        persist.assert_not_awaited()
        assert runner.restore == {"model": "standing"}
    else:
        persist.assert_awaited_once()
    store.close_all_db_handles()
    restarted = SessionStore(home / "sessions", GatewayConfig())
    assert restarted.get_execution_profile(key) == "default"
    assert restarted.get_model_override(key) == ({"model": "standing"} if one_turn else None)
    restarted.close_all_db_handles()


_TITLE_CONFIG = {"gateway": {"spawn": {
    "parent_channel_id": "forum-1",
    "models": {
        "sol-high": {"label": "Sol High", "model": "gpt-5.6-sol", "provider": "openai-codex",
                     "reasoning_effort": "high"},
        "sol-xhigh": {"label": "Sol XHigh", "model": "gpt-5.6-sol", "provider": "openai-codex",
                      "reasoning_effort": "xhigh"},
    },
}}}


class _TitleAdapter:
    def __init__(self, thread):
        self.renames = []
        self._client = SimpleNamespace(get_channel=lambda _id: None, fetch_channel=AsyncMock(return_value=thread))

    async def rename_thread(self, thread_id, name, *, only_if_current_name=None):
        self.renames.append((thread_id, name, only_if_current_name))
        return True


class _TitleRunner:
    _is_discord_spawn_forum_lane = GatewayRunner._is_discord_spawn_forum_lane
    _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
    _is_relay_discord_channel_lane = GatewayRunner._is_relay_discord_channel_lane
    _spawn_forum_title_rename = GatewayRunner._spawn_forum_title_rename
    _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title
    _rename_discord_auto_thread_for_session_title = GatewayRunner._rename_discord_auto_thread_for_session_title
    _session_key_for_source = GatewayRunner._session_key_for_source

    def __init__(self, adapter, override):
        self.adapters = {Platform.DISCORD: adapter}
        self.async_session_store = SimpleNamespace(get_model_override=AsyncMock(return_value=override))
        self.config = GatewayConfig()

    def _delivery_adapter_for(self, source):
        return self.adapters[source.platform]


def _forum_source(parent="forum-1"):
    return SessionSource(platform=Platform.DISCORD, chat_id="12345", chat_type="thread", thread_id="12345",
                         parent_chat_id=parent, user_id="owner-1")


async def _semantic_rename(thread, title="Fix dot title inference", source=None):
    adapter = _TitleAdapter(thread)
    runner = _TitleRunner(adapter, {"model": "gpt-5.6-sol", "provider": "openai-codex"})
    with patch("gateway.run._load_gateway_config", return_value=_TITLE_CONFIG):
        await runner._rename_discord_auto_thread_for_session_title(source or _forum_source(), "session-1", title)
    return adapter.renames


@pytest.mark.asyncio
async def test_placeholder_forum_topic_takes_one_guarded_semantic_rename_with_tagged_label():
    thread = SimpleNamespace(name=".", applied_tags=[SimpleNamespace(name="sol-xhigh")])

    # Both aliases share the wire model and the persisted override has no effort: the post's model tag decides.
    assert await _semantic_rename(thread) == [("12345", "Fix dot title inference · Sol XHigh", ".")]


@pytest.mark.asyncio
async def test_hearted_placeholder_keeps_heart_through_semantic_rename():
    thread = SimpleNamespace(name="❤️ New chat · Sol High", applied_tags=[SimpleNamespace(name="sol-high")])

    assert await _semantic_rename(thread) == [
        ("12345", "❤️ Fix dot title inference · Sol High", "❤️ New chat · Sol High"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Benchmark flaky auth · Sol High", "my own words", None])
async def test_human_title_or_unknown_name_is_never_renamed(name):
    thread = SimpleNamespace(name=name, applied_tags=[SimpleNamespace(name="sol-high")])

    assert await _semantic_rename(thread) == []


@pytest.mark.asyncio
async def test_thread_outside_spawn_forum_is_not_a_semantic_rename_lane():
    thread = SimpleNamespace(name=".", applied_tags=[])

    assert await _semantic_rename(thread, source=_forum_source(parent="other-channel")) == []


def test_title_callback_schedules_semantic_rename_only_in_spawn_forum():
    from gateway.run_turn_runner import TurnRunner

    scheduled = []

    class LaneRunner:
        _is_telegram_topic_lane = lambda self, source: False  # noqa: E731
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _is_relay_discord_channel_lane = GatewayRunner._is_relay_discord_channel_lane
        _is_discord_spawn_forum_lane = GatewayRunner._is_discord_spawn_forum_lane

        def _schedule_discord_semantic_thread_rename(self, source, session_id, title):
            scheduled.append((source.thread_id, session_id, title))

    holder = SimpleNamespace(_runner=LaneRunner())

    def attach(source):
        agent = SimpleNamespace(session_id="sess-1", _session_db=None)
        TurnRunner._attach_session_title_callback(holder, agent, SimpleNamespace(source=source))
        return getattr(agent, "_on_session_title", None)

    with patch("gateway.run._load_gateway_config", return_value=_TITLE_CONFIG):
        callback = attach(_forum_source())
        assert attach(_forum_source(parent="other-channel")) is None
    callback("fix the dot title", "derived")
    callback("Fix dot title inference", "llm")

    assert scheduled == [("12345", "sess-1", "Fix dot title inference")]


@pytest.mark.parametrize(("name", "placeholder"), [
    (".", True), ("❤️ .", True), ("New chat · Sol High", True), ("untitled", True), ("—", True),
    ("Fix dot title · Sol High", False), ("main · Sol High", False), ("❤️ Research plan", False),
])
def test_spawn_topic_placeholder_boundary(name, placeholder):
    from gateway.spawn_command import _is_topic_placeholder

    assert _is_topic_placeholder(name) is placeholder
