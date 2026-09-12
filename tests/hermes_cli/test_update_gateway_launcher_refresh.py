"""Legacy pythonw launcher normalization + post-update launcher refresh.

Covers the two halves of the "legacy pythonw gateways survive updates
forever" gap:

1. ``gateway_windows._resolve_detached_python`` — normalizes a legacy
   ``pythonw.exe`` interpreter (pre-aa2ae36c3f launchers / argv snapshots)
   to the sibling console ``python.exe`` so respawns and regenerated
   launchers use the hidden-console design (#54220/#56747) and don't die
   with ``RuntimeError: sys.stderr is None`` (#71671).
2. ``cli_main._refresh_windows_gateway_launchers`` — ``hermes
   update`` regenerates the installed Scheduled Task / Startup launcher
   scripts instead of leaving install-time artifacts stale forever.

``_resolve_detached_python`` is a pure path helper and runs on any host.
``windowless_gateway_restart_spec`` returns its argv unchanged off Windows,
so the test that exercises the rewrite is ``windows_only`` rather than run
against a faked ``sys.platform``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as cli_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# _resolve_detached_python: legacy pythonw normalization
# ---------------------------------------------------------------------------


def _make_venv(tmp_path: Path, *, with_console_python: bool) -> tuple[Path, Path]:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    pythonw = scripts / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    python = scripts / "python.exe"
    if with_console_python:
        python.write_text("", encoding="utf-8")
    return pythonw, python


def test_resolve_detached_python_swaps_legacy_pythonw_for_console_sibling(tmp_path):
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    exe, venv_dir, extra = gateway_windows._resolve_detached_python(str(pythonw))

    assert exe == str(python)
    assert venv_dir == tmp_path / "venv"
    assert extra == []


def _make_stable_windows_venv(project_root: Path) -> tuple[Path, Path]:
    python = project_root / "venv" / "Scripts" / "python.exe"
    bundle = project_root / "venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
    python.parent.mkdir(parents=True)
    bundle.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    bundle.write_text("test-ca", encoding="utf-8")
    return python, bundle


def test_stable_launcher_python_replaces_disposable_update_runner(tmp_path):
    project_root = tmp_path / "install" / "hermes-agent"
    stable_python, _ = _make_stable_windows_venv(project_root)
    transient_python = tmp_path / "Temp" / "hermes-update-runner" / "Scripts" / "python.exe"

    with mock.patch("hermes_cli.gateway.get_python_path", return_value=str(transient_python)):
        selected = gateway_windows._stable_launcher_python_path(project_root)

    assert selected == str(stable_python)


def test_stable_launcher_python_preserves_external_venv(tmp_path):
    project_root = tmp_path / "install" / "hermes-agent"
    _make_stable_windows_venv(project_root)
    external_python = tmp_path / "durable-venvs" / "hermes" / "Scripts" / "python.exe"

    with mock.patch("hermes_cli.gateway.get_python_path", return_value=str(external_python)):
        selected = gateway_windows._stable_launcher_python_path(project_root)

    assert selected == str(external_python)


def test_ca_overlay_remaps_only_disposable_update_runner_values(monkeypatch, tmp_path):
    project_root = tmp_path / "install" / "hermes-agent"
    _, stable_bundle = _make_stable_windows_venv(project_root)
    transient_bundle = tmp_path / "Temp" / "hermes-update-runner" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
    corporate_bundle = tmp_path / "company" / "root-ca.pem"
    monkeypatch.setenv("SSL_CERT_FILE", str(transient_bundle))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(corporate_bundle))
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)

    overlay = gateway_windows._stable_ca_bundle_overlay(project_root)

    assert overlay == {"SSL_CERT_FILE": str(stable_bundle)}


def test_ca_overlay_is_empty_without_disposable_update_runner(monkeypatch, tmp_path):
    project_root = tmp_path / "install" / "hermes-agent"
    _make_stable_windows_venv(project_root)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "company" / "root-ca.pem"))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)

    assert gateway_windows._stable_ca_bundle_overlay(project_root) == {}




@pytest.mark.windows_only
def test_restart_spec_normalizes_legacy_pythonw_argv(tmp_path):
    """A pre-rework Scheduled Task argv snapshot (leading pythonw.exe) must be
    respawned through the console python + hidden-console launch, with every
    argument after the interpreter preserved verbatim.

    ``windows_only``: ``windowless_gateway_restart_spec`` returns the argv
    untouched off Windows, so the fake was the only thing making the rewrite
    (and its ``Scripts/``-layout venv derivation) run at all.
    """
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    argv = [str(pythonw), "-m", "hermes_cli.main", "gateway", "run"]
    with mock.patch.object(
        gateway_windows, "_stable_gateway_working_dir", return_value=str(tmp_path)
    ), mock.patch("hermes_cli.config.get_hermes_home", return_value=str(tmp_path)):
        new_argv, cwd, env = gateway_windows.windowless_gateway_restart_spec(list(argv))

    assert new_argv[0] == str(python)
    assert new_argv[1:] == argv[1:]
    assert cwd == str(tmp_path)
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")


# ---------------------------------------------------------------------------
# _refresh_windows_gateway_launchers: hermes update regenerates launchers
# ---------------------------------------------------------------------------








