"""Regression tests for gateway per-turn env reload preserving config authority.

Issue #19158: startup bridges config.yaml agent.max_turns into
HERMES_MAX_ITERATIONS, but a later per-turn load_dotenv(..., override=True)
can restore a stale .env HERMES_MAX_ITERATIONS value before the next turn.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from gateway import run as gateway_run


def test_reload_runtime_env_preserves_config_max_turns(tmp_path: Path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"max_turns": 9000}}),
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "HERMES_MAX_ITERATIONS=90\nOPENROUTER_API_KEY=fresh-key\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "9000")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["OPENROUTER_API_KEY"] == "fresh-key"
    assert os.environ["HERMES_MAX_ITERATIONS"] == "9000"


def test_reload_runtime_env_preserves_config_terminal_backend(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for #29186: the per-turn .env reload must not restore a
    stale TERMINAL_ENV=docker over config.yaml's terminal.backend=local.

    This is the exact mid-session backend flip from the field report: the
    gateway starts on the bridged local backend, works for hours, then a
    later turn's reload re-loads .env with override=True and every terminal /
    execute_code / read_file call starts trying Docker — while
    ``hermes config get terminal.backend`` still says local.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local"}}),
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Startup bridge already ran: the effective backend is local.
    monkeypatch.setenv("TERMINAL_ENV", "local")

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["TERMINAL_ENV"] == "local"


def test_current_max_iterations_reloads_before_reading(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "90")

    def _fake_reload() -> None:
        os.environ["HERMES_MAX_ITERATIONS"] = "200"

    monkeypatch.setattr(
        gateway_run,
        "_reload_runtime_env_preserving_config_authority",
        _fake_reload,
    )

    assert gateway_run._current_max_iterations() == 200


def test_profile_scoped_reload_does_not_mutate_process_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    """A brokered /spawn turn must not leak worker secrets into Main."""
    from agent.secret_scope import current_secret_scope, get_secret

    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"max_turns": 321}}),
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "OPENROUTER_API_KEY=worker-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "main-secret")

    with gateway_run._profile_runtime_scope(profile_home):
        assert current_secret_scope() is not None
        assert get_secret("OPENROUTER_API_KEY") == "worker-secret"
        gateway_run._reload_runtime_env_preserving_config_authority()
        assert os.environ["OPENROUTER_API_KEY"] == "main-secret"
        assert get_secret("OPENROUTER_API_KEY") == "worker-secret"

    assert current_secret_scope() is None
    assert os.environ["OPENROUTER_API_KEY"] == "main-secret"
