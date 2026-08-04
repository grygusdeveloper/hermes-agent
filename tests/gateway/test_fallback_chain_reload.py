"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import time
import threading
from types import SimpleNamespace


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_refresh_fallback_model_clears_when_config_removed(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=[{"provider": "stale", "model": "x"}],
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    assert bound() is not None

    cfg.write_text("model:\n  provider: nvidia\n")
    assert bound() is None
    assert runner._fallback_model is None


def test_profile_scope_empty_fallback_chain_overrides_main_chain(tmp_path, monkeypatch):
    """A spawned execution profile with ``fallback_providers: []`` must not
    inherit Main's fallback chain on either fresh creation or cached-agent reuse.
    """
    from gateway.run import GatewayRunner

    main_home = tmp_path / "main"
    profile_home = tmp_path / "profiles" / "gemini"
    main_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    (main_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: zai\n"
        "    model: glm-5.2\n",
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        "fallback_providers: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._hermes_home", main_home)
    monkeypatch.setattr(
        "gateway.run.get_hermes_home_override", lambda: profile_home
    )

    assert GatewayRunner._load_fallback_model() is None
    runner = SimpleNamespace(
        _fallback_model=[{"provider": "zai", "model": "glm-5.2"}],
    )
    refreshed = GatewayRunner._refresh_fallback_model.__get__(runner)()
    assert refreshed is None
    assert runner._fallback_model is None

    cached_agent = SimpleNamespace(
        _fallback_chain=[{"provider": "zai", "model": "glm-5.2"}],
        _fallback_model={"provider": "zai", "model": "glm-5.2"},
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
    )
    GatewayRunner._apply_fallback_chain_to_agent(cached_agent, refreshed)
    assert cached_agent._fallback_chain == []
    assert cached_agent._fallback_model is None


def test_refresh_fallback_model_keeps_last_known_good_on_read_failure(
    tmp_path, monkeypatch,
):
    """A transient config.yaml read/parse failure (user mid-edit, non-atomic
    write) must NOT wipe the last known-good chain — only a successful read
    that genuinely lacks the key clears it."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    good = bound()
    assert good == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]

    # Simulate a mid-edit torn write: invalid YAML.
    cfg.write_text("fallback_providers:\n  - provider: [unclosed\n")
    assert bound() == good
    assert runner._fallback_model == good


def test_transient_profile_failure_never_borrows_main_last_known_good(
    tmp_path, monkeypatch,
):
    """A profile cache miss fails closed even when Main populated the
    compatibility mirror with a non-empty fallback chain."""
    from gateway.run import GatewayRunner

    main_home = tmp_path / "main"
    profile_home = tmp_path / "profiles" / "gemini"
    main_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    (main_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: zai\n"
        "    model: glm-5.2\n",
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        "fallback_providers:\n  - provider: [unclosed\n",
        encoding="utf-8",
    )
    active_home = {"value": main_home}
    monkeypatch.setattr(
        "gateway.run.get_hermes_home_override",
        lambda: active_home["value"],
    )
    runner = SimpleNamespace(
        _fallback_model=None,
        _fallback_models_by_config_home={},
    )
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)

    main_chain = bound()
    assert main_chain == [{"provider": "zai", "model": "glm-5.2"}]
    assert runner._fallback_model == main_chain

    active_home["value"] = profile_home
    assert bound() is None
    assert runner._fallback_model is None
    assert runner._fallback_models_by_config_home[
        str((main_home / "config.yaml").resolve())
    ] == main_chain


def test_concurrent_profile_refreshes_keep_last_known_good_isolated(
    tmp_path, monkeypatch,
):
    """Concurrent transient failures return each profile's own cached chain."""
    from gateway.run import GatewayRunner

    homes = {
        "main": tmp_path / "main",
        "gemini": tmp_path / "profiles" / "gemini",
    }
    chains = {
        "main": [{"provider": "zai", "model": "glm-5.2"}],
        "gemini": [{"provider": "openrouter", "model": "safe-profile-model"}],
    }
    for name, home in homes.items():
        home.mkdir(parents=True)
        entry = chains[name][0]
        (home / "config.yaml").write_text(
            "fallback_providers:\n"
            f"  - provider: {entry['provider']}\n"
            f"    model: {entry['model']}\n",
            encoding="utf-8",
        )

    local = threading.local()
    monkeypatch.setattr(
        "gateway.run.get_hermes_home_override",
        lambda: local.home,
    )
    runner = SimpleNamespace(
        _fallback_model=None,
        _fallback_models_by_config_home={},
    )
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    ready = threading.Barrier(2)
    outcomes: dict[str, list] = {}

    def worker(name: str) -> None:
        local.home = homes[name]
        assert bound() == chains[name]
        (homes[name] / "config.yaml").write_text(
            "fallback_providers:\n  - provider: [unclosed\n",
            encoding="utf-8",
        )
        ready.wait(timeout=2)
        outcomes[name] = [bound() for _ in range(20)]

    threads = [threading.Thread(target=worker, args=(name,)) for name in homes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert outcomes["main"] == [chains["main"]] * 20
    assert outcomes["gemini"] == [chains["gemini"]] * 20


def test_apply_fallback_chain_updates_primary_agent():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
    )
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, chain)

    assert agent._fallback_chain == chain
    assert agent._fallback_model == chain[0]
    assert agent._fallback_index == 0


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True


def test_background_and_main_agent_paths_call_refresh():
    """Both AIAgent construction sites must pass a refreshed chain, not the
    startup snapshot, and the cached-agent reuse path must apply the refreshed
    chain. Source-level invariant for call sites that resist unit testing.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent.parent / "gateway" / "run.py"
    ).read_text(encoding="utf-8")
    # The agent-construction site inside TurnRunner.run_sync (extracted from
    # the old _run_agent_inner closure) references the runner as
    # ``self._runner``; the background-agent site still uses bare ``self``.
    _refresh_calls = (
        source.count("fallback_model=self._refresh_fallback_model()")
        + source.count("fallback_model=self._runner._refresh_fallback_model()")
    )
    assert _refresh_calls >= 2
    # The cached-agent reuse path (the load-bearing fix for a long-lived
    # session in a running gateway) must apply the refreshed chain.
    assert (
        "self._apply_fallback_chain_to_agent(" in source
        or "self._runner._apply_fallback_chain_to_agent(" in source
    )
    # The stale startup-snapshot form must not remain at create sites.
    assert "fallback_model=self._fallback_model," not in source
    assert "fallback_model=self._runner._fallback_model," not in source


def test_load_fallback_model_static_unchanged_contract(tmp_path, monkeypatch):
    """_load_fallback_model remains a pure static reader used by refresh."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "fallback_model:\n"
        "  provider: nous\n"
        "  model: Hermes-4\n"
    )

    chain = GatewayRunner._load_fallback_model()
    assert chain == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "nous", "model": "Hermes-4"},
    ]
