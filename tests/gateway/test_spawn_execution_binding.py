"""A brokered thread keeps Main's route while its execution binding survives a restart."""

from pathlib import Path
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


def test_spawn_binding_persists_across_reset_and_reload(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-1", thread_id="thread-1",
                           chat_type="thread", parent_chat_id="main-forum")
    path = home / "sessions"
    store = SessionStore(path, GatewayConfig())
    key = store.get_or_create_session(source).session_key
    assert key.startswith("agent:main:")
    store.set_execution_profile(key, "Researcher")
    assert store.get_execution_profile(key) == "researcher"
    with pytest.raises(ValueError):
        store.set_execution_profile(key, "../../default")
    assert store.get_execution_profile(key) == "researcher"
    store.reset_session(key)
    assert store.get_execution_profile(key) == "researcher"
    store.close_all_db_handles()
    restarted = SessionStore(path, GatewayConfig())
    assert restarted.get_execution_profile(key) == "researcher"
    assert restarted.get_or_create_session(source).session_key == key
    restarted.close_all_db_handles()


@pytest.mark.asyncio
async def test_brokered_turn_scopes_credentials_without_rerouting_delivery(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    satellite = home / "profiles" / "researcher"
    satellite.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".env").write_text("SPAWN_SCOPE_CHECK=main\n")
    (satellite / ".env").write_text("SPAWN_SCOPE_CHECK=researcher\n")
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-1", thread_id="thread-1",
                           chat_type="thread", parent_chat_id="main-forum")
    store = SessionStore(home / "sessions", GatewayConfig())
    key = store.get_or_create_session(source).session_key
    store.set_execution_profile(key, "researcher")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    seen = []

    async def execute(*args, **kwargs):
        from agent.secret_scope import get_secret
        seen.append((get_hermes_home(), get_secret("SPAWN_SCOPE_CHECK"), source.profile, key))
        return {"completed": True}

    runner._run_agent_inner = execute
    assert (await runner._run_agent("task", "", [], source, "s1", session_key=key))["completed"]
    store.set_execution_profile(key, None)
    assert (await runner._run_agent("task", "", [], source, "s1", session_key=key))["completed"]
    store.set_execution_profile(key, "researcher")
    assert (await runner._run_agent("task", "", [], source, "s1", session_key=key))["completed"]
    assert [item[0] for item in seen] == [satellite, home, satellite]
    assert [item[1] for item in seen] == ["researcher", None, "researcher"]
    assert all(item[2] is None and item[3] == key for item in seen)
    store.close_all_db_handles()
    runner.session_store = SessionStore(home / "sessions", GatewayConfig())
    assert (await runner._run_agent("follow-up", "", [], source, "s2", session_key=key))["completed"]
    assert seen[-1][:2] == (satellite, "researcher")
    satellite.rename(home / "profiles" / "researcher-offline")
    refused = await runner._run_agent("blocked", "", [], source, "s2", session_key=key)
    assert refused["completed"] is False
    assert len(seen) == 4
    runner.session_store.close_all_db_handles()
