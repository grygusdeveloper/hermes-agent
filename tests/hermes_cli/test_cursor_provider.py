"""First-class Cursor provider/auth regressions (all subprocesses mocked)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.copilot_acp_client import (
    CopilotACPClient,
    CURSOR_MARKER_BASE_URL,
    _build_subprocess_env,
    _copilot_deprecation_error_message,
    is_acp_stdio_runtime,
)
from agent.auxiliary_client import _normalize_aux_provider, resolve_provider_client
from hermes_cli import cursor_cli
from hermes_cli.auth import (
    AuthError,
    PROVIDER_REGISTRY,
    get_auth_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)
from hermes_cli.auth_commands import (
    _normalize_provider,
    auth_add_command,
    auth_logout_command,
    auth_status_command,
)
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import CANONICAL_PROVIDERS, normalize_provider, provider_model_ids
from hermes_cli.providers import HERMES_OVERLAYS
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli import model_switch
from providers import get_provider_profile


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_cursor_has_distinct_first_class_identity():
    assert PROVIDER_REGISTRY["cursor"].auth_type == "external_process"
    assert PROVIDER_REGISTRY["cursor"].inference_base_url == "acp://cursor"
    assert HERMES_OVERLAYS["cursor"].base_url_override == "acp://cursor"
    assert normalize_provider("cursor-agent") == "cursor"
    assert normalize_provider("cursor-cli") == "cursor"
    assert _normalize_aux_provider("cursor-agent") == "cursor"
    assert _normalize_aux_provider("cursor-cli") == "cursor"
    assert any(entry.slug == "cursor" for entry in CANONICAL_PROVIDERS)
    assert normalize_provider("github-copilot-acp") == "copilot-acp"
    assert resolve_provider("cursor-agent") == "cursor"
    assert resolve_provider("cursor-cli") == "cursor"
    assert _normalize_provider("cursor-agent") == "cursor"
    assert _normalize_provider("cursor-cli") == "cursor"
    profile = get_provider_profile("cursor")
    assert profile is not None and profile.name == "cursor"
    alias_profile = get_provider_profile("cursor-agent")
    assert alias_profile is not None and alias_profile.name == "cursor"
    assert is_acp_stdio_runtime(provider="cursor", base_url="acp://cursor")
    assert is_acp_stdio_runtime(provider="cursor-agent", base_url="")
    assert is_acp_stdio_runtime(provider="cursor-cli", base_url="")
    assert is_acp_stdio_runtime(provider="custom", base_url="acp://cursor")
    assert is_acp_stdio_runtime(provider="copilot-acp", base_url="acp://copilot")
    assert is_acp_stdio_runtime(provider="github-copilot-acp", base_url="")
    assert is_acp_stdio_runtime(provider="custom", base_url="acp://cursor/")
    assert is_acp_stdio_runtime(provider="custom", base_url="acp://copilot/")
    assert not is_acp_stdio_runtime(provider="custom", base_url="acp://cursor-evil")
    assert not is_acp_stdio_runtime(provider="custom", base_url="acp://copilot-extra")
    assert not is_acp_stdio_runtime(provider="claude-code", base_url="acp://claude-code")
    assert not is_acp_stdio_runtime(provider="xai", base_url="https://api.x.ai/v1")
    assert not is_acp_stdio_runtime(provider="gemini", base_url="acp://antigravity")


def test_cursor_model_ids_are_forwarded_exactly():
    assert normalize_model_for_provider("cursor-grok-4.6-high", "cursor") == "cursor-grok-4.6-high"
    assert normalize_model_for_provider("cursor-grok-4.6-xhigh-fast", "cursor") == "cursor-grok-4.6-xhigh-fast"
    assert normalize_model_for_provider("cursor-grok-4.6-high", "cursor-agent") == "cursor-grok-4.6-high"
    # Cursor IDs are account-specific exact strings; do not strip vendor prefixes.
    assert (
        normalize_model_for_provider("x-ai/cursor-grok-4.6-high", "cursor")
        == "x-ai/cursor-grok-4.6-high"
    )
    assert cursor_cli.cursor_acp_args("cursor-grok-4.6-high") == [
        "--model",
        "cursor-grok-4.6-high",
        "acp",
    ]


def test_model_parser_is_strict_and_deduplicates():
    text = """cursor-grok-4.6-high - Grok High
cursor-grok-4.6-high - duplicate
--evil - rejected
bad model - rejected
cursor-grok-4.6-xhigh-fast - Grok XHigh Fast
"""
    assert cursor_cli.parse_cursor_models(text) == [
        "cursor-grok-4.6-high",
        "cursor-grok-4.6-xhigh-fast",
    ]
    with pytest.raises(cursor_cli.CursorCLIError):
        cursor_cli.cursor_acp_args("--model bad")


def test_status_is_live_secret_free_and_exact(monkeypatch):
    monkeypatch.setattr(cursor_cli.shutil, "which", lambda _command: "/opt/cursor/agent")
    monkeypatch.setattr(
        cursor_cli.subprocess,
        "run",
        lambda argv, **kwargs: _Completed(
            stdout=json.dumps(
                {
                    "isAuthenticated": True,
                    "email": "private@example.test",
                    "accessToken": "do-not-return-this",
                }
            )
        ),
    )
    status = cursor_cli.get_cursor_auth_status(command="agent")
    assert status["logged_in"] is True
    assert status["resolved_command"] == "/opt/cursor/agent"
    blob = json.dumps(status)
    assert "private@example.test" not in blob
    assert "do-not-return-this" not in blob


def test_status_reports_malformed_json_missing_cli_and_timeout(monkeypatch):
    monkeypatch.setattr(cursor_cli.shutil, "which", lambda _command: "/opt/cursor/agent")
    monkeypatch.setattr(cursor_cli.subprocess, "run", lambda *a, **k: _Completed(stdout="not-json"))
    assert cursor_cli.get_cursor_auth_status(command="agent")["error_code"] == "cursor_status_malformed"

    monkeypatch.setattr(cursor_cli.shutil, "which", lambda _command: None)
    assert cursor_cli.get_cursor_auth_status(command="agent")["error_code"] == "missing_cursor_cli"

    monkeypatch.setattr(cursor_cli.shutil, "which", lambda _command: "/opt/cursor/agent")
    monkeypatch.setattr(
        cursor_cli.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("agent", 1)),
    )
    assert cursor_cli.get_cursor_auth_status(command="agent")["error_code"] == "cursor_status_timeout"


def test_login_and_logout_use_supported_commands_and_verify(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-reach-cursor")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/copilot")
    monkeypatch.setattr(cursor_cli, "resolve_cursor_command", lambda **_k: ("agent", "/opt/cursor/agent"))
    statuses = iter([
        {"logged_in": False},
        {"logged_in": True},
        {"logged_in": False},
    ])
    monkeypatch.setattr(cursor_cli, "get_cursor_auth_status", lambda **_k: next(statuses))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return _Completed()

    monkeypatch.setattr(cursor_cli.subprocess, "run", fake_run)
    assert cursor_cli.login_cursor(no_browser=True)["logged_in"] is True
    assert cursor_cli.logout_cursor()["logged_in"] is False
    assert calls[0][0] == ["/opt/cursor/agent", "login"]
    assert calls[0][1]["NO_OPEN_BROWSER"] == "1"
    assert "OPENAI_API_KEY" not in calls[0][1]
    assert "HERMES_COPILOT_ACP_COMMAND" not in calls[0][1]
    assert calls[1][0] == ["/opt/cursor/agent", "logout"]


def test_runtime_resolves_exact_model_and_no_real_secret(monkeypatch):
    monkeypatch.setattr(cursor_cli, "resolve_cursor_command", lambda **_k: ("agent", "/opt/cursor/agent"))
    monkeypatch.setattr(cursor_cli, "get_cursor_auth_status", lambda **_k: {"logged_in": True})
    creds = resolve_external_process_provider_credentials(
        "cursor", target_model="cursor-grok-4.6-high"
    )
    runtime = resolve_runtime_provider(
        requested="cursor", target_model="cursor-grok-4.6-high"
    )
    expected_args = ["--model", "cursor-grok-4.6-high", "acp"]
    assert creds["provider"] == runtime["provider"] == "cursor"
    assert creds["base_url"] == runtime["base_url"] == CURSOR_MARKER_BASE_URL
    assert creds["args"] == runtime["args"] == expected_args
    assert creds["command"] == runtime["command"] == "/opt/cursor/agent"
    alias_creds = resolve_external_process_provider_credentials(
        "cursor-agent", target_model="cursor-grok-4.6-high"
    )
    assert alias_creds["provider"] == "cursor"
    assert alias_creds["args"] == expected_args
    assert "token" not in json.dumps(runtime).lower()


def test_model_discovery_uses_live_result_then_safe_fallback(monkeypatch):
    monkeypatch.setattr(cursor_cli, "discover_cursor_models", lambda: ["cursor-new-model"])
    assert provider_model_ids("cursor") == ["cursor-new-model"]
    monkeypatch.setattr(
        cursor_cli,
        "discover_cursor_models",
        lambda: (_ for _ in ()).throw(cursor_cli.CursorCLIError("offline")),
    )
    assert "cursor-grok-4.6-high" in provider_model_ids("cursor")


def test_authenticated_cursor_appears_in_model_picker(monkeypatch):
    from hermes_cli import auth, providers

    cursor_overlay = providers.HERMES_OVERLAYS["cursor"]
    monkeypatch.setattr(providers, "HERMES_OVERLAYS", {"cursor": cursor_overlay})
    monkeypatch.setattr(
        auth,
        "get_auth_status",
        lambda provider: {"logged_in": provider == "cursor"},
    )
    from hermes_cli import models

    monkeypatch.setattr(
        models,
        "cached_provider_model_ids",
        lambda provider: ["cursor-grok-4.6-high"] if provider == "cursor" else [],
    )
    rows = model_switch.list_authenticated_providers(
        max_models=None,
        probe_custom_providers=False,
    )
    cursor_rows = [row for row in rows if row["slug"] == "cursor"]
    assert len(cursor_rows) == 1
    assert cursor_rows[0]["models"] == ["cursor-grok-4.6-high"]


def test_cursor_acp_child_strips_unrelated_provider_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-reach-cursor")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-reach-cursor")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")
    cursor_env = _build_subprocess_env("acp://cursor")
    copilot_env = _build_subprocess_env("acp://copilot")
    antigravity_env = _build_subprocess_env("acp://antigravity")
    assert "OPENAI_API_KEY" not in cursor_env
    assert "ANTHROPIC_API_KEY" not in cursor_env
    assert "HERMES_COPILOT_ACP_COMMAND" not in cursor_env
    assert copilot_env.get("OPENAI_API_KEY") == "should-not-reach-cursor"
    assert copilot_env.get("HERMES_COPILOT_ACP_COMMAND") == "/opt/copilot"
    assert antigravity_env.get("OPENAI_API_KEY") == "should-not-reach-cursor"


def test_cursor_status_and_acp_children_share_real_home(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-reach-cursor")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/copilot")
    monkeypatch.setattr(
        cursor_cli,
        "get_real_home",
        lambda env=None: "/real/os-home",
    )
    monkeypatch.setattr(
        "hermes_constants.get_real_home",
        lambda env=None: "/real/os-home",
    )
    status_env = cursor_cli._cursor_child_env()
    acp_env = _build_subprocess_env("acp://cursor")
    assert status_env["HOME"] == acp_env["HOME"] == "/real/os-home"
    assert "OPENAI_API_KEY" not in status_env
    assert "OPENAI_API_KEY" not in acp_env
    assert "HERMES_COPILOT_ACP_COMMAND" not in status_env
    assert "HERMES_COPILOT_ACP_COMMAND" not in acp_env


def test_cursor_does_not_emit_copilot_deprecation_guidance():
    stderr = "The gh-copilot extension has been deprecated. No commands will be executed."
    assert _copilot_deprecation_error_message(stderr, base_url="acp://cursor") is None
    copilot_msg = _copilot_deprecation_error_message(stderr, base_url="acp://copilot")
    assert copilot_msg is not None
    assert "HERMES_COPILOT_ACP_COMMAND" in copilot_msg


def test_cursor_client_refuses_copilot_env_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")
    with pytest.raises(ValueError, match="explicit command"):
        CopilotACPClient(base_url="acp://cursor")

    client = CopilotACPClient(
        api_key="cursor-agent",
        base_url="acp://cursor",
        command="/opt/cursor/agent",
        args=["--model", "cursor-grok-4.6-high", "acp"],
    )
    assert client._acp_command == "/opt/cursor/agent"
    assert client._acp_args == ["--model", "cursor-grok-4.6-high", "acp"]
    assert client.api_key == "cursor-agent"

    copilot = CopilotACPClient(base_url="acp://copilot")
    assert copilot._acp_command == "/opt/copilot"
    assert copilot._acp_args == ["--acp", "--stdio"]


def test_cursor_client_sentinel_key_without_base_url_does_not_use_copilot_env(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/opt/copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")
    with pytest.raises(ValueError, match="explicit command"):
        CopilotACPClient(api_key="cursor-agent")

    client = CopilotACPClient(
        api_key="cursor-agent",
        command="/opt/cursor/agent",
        args=["--model", "cursor-grok-4.6-high", "acp"],
    )
    assert client.base_url == CURSOR_MARKER_BASE_URL
    assert client._acp_command == "/opt/cursor/agent"
    assert client._acp_args == ["--model", "cursor-grok-4.6-high", "acp"]
    env = _build_subprocess_env(client.base_url)
    assert "HERMES_COPILOT_ACP_COMMAND" not in env
    assert "HERMES_COPILOT_ACP_ARGS" not in env

    copilot = CopilotACPClient(api_key="cursor-agent", base_url="acp://copilot")
    assert copilot.base_url == "acp://copilot"
    assert copilot._acp_command == "/opt/copilot"


def test_cursor_command_reads_top_level_then_providers_block(monkeypatch):
    monkeypatch.setattr(
        cursor_cli.shutil,
        "which",
        lambda command: command if str(command).startswith("/") else f"/usr/bin/{command}",
    )
    top_level = cursor_cli.resolve_cursor_command({"cursor": {"command": "/opt/top/agent"}})
    assert top_level == ("/opt/top/agent", "/opt/top/agent")
    from_providers = cursor_cli.resolve_cursor_command(
        {"cursor": {"command": ""}, "providers": {"cursor": {"command": "/opt/providers/agent"}}}
    )
    assert from_providers == ("/opt/providers/agent", "/opt/providers/agent")
    both = cursor_cli.resolve_cursor_command(
        {
            "cursor": {"command": "/opt/top/agent"},
            "providers": {"cursor": {"command": "/opt/providers/agent"}},
        }
    )
    assert both[0] == "/opt/top/agent"


def test_cursor_command_schema_is_known():
    from hermes_cli.config import DEFAULT_CONFIG, _validate_config_key

    assert DEFAULT_CONFIG["cursor"]["command"] == ""
    known, _suggestion = _validate_config_key("cursor.command")
    assert known is True


def test_acp_error_text_is_redacted():
    from agent.copilot_acp_client import _safe_acp_error_text

    leaked = "ACP failed token=ghp_abcdefghijklmnopqrstuvwxyz012345"
    cleaned = _safe_acp_error_text(leaked)
    assert "ghp_abcdefghijklmnopqrstuvwxyz012345" not in cleaned


def test_shell_fragment_command_is_rejected():
    with pytest.raises(cursor_cli.CursorCLIError) as exc:
        cursor_cli.resolve_cursor_command({"cursor": {"command": "agent; rm -rf /"}})
    assert exc.value.code == "invalid_cursor_command"


def test_missing_model_becomes_auth_error(monkeypatch):
    monkeypatch.setattr(cursor_cli, "resolve_cursor_command", lambda **_k: ("agent", "/opt/cursor/agent"))
    monkeypatch.setattr(cursor_cli, "get_cursor_auth_status", lambda **_k: {"logged_in": True})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    with pytest.raises(AuthError) as exc:
        resolve_external_process_provider_credentials("cursor")
    assert exc.value.code == "invalid_cursor_model"


def test_get_auth_status_dispatches_to_live_cursor_status(monkeypatch):
    monkeypatch.setattr(
        cursor_cli,
        "get_cursor_auth_status",
        lambda **_k: {
            "provider": "cursor",
            "logged_in": True,
            "auth_type": "cursor_cli",
            "command": "agent",
        },
    )
    status = get_auth_status("cursor")
    assert status["logged_in"] is True
    assert status["auth_type"] == "cursor_cli"
    assert "token" not in json.dumps(status).lower()
    alias_status = get_auth_status("cursor-agent")
    assert alias_status["logged_in"] is True
    assert alias_status["auth_type"] == "cursor_cli"
    assert get_auth_status("cursor-cli")["provider"] == "cursor"


def test_auxiliary_cursor_forwards_exact_command_and_args(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda provider, **kw: {
            "provider": "cursor",
            "api_key": "cursor-agent",
            "base_url": "acp://cursor",
            "command": "/opt/cursor/agent",
            "args": ["--model", kw.get("target_model"), "acp"],
            "source": "cursor-cli",
        },
    )
    monkeypatch.setattr("agent.copilot_acp_client.CopilotACPClient", _FakeClient)
    client, model = resolve_provider_client("cursor", "cursor-grok-4.6-high")
    assert model == "cursor-grok-4.6-high"
    assert isinstance(client, _FakeClient)
    assert captured["command"] == "/opt/cursor/agent"
    assert captured["args"] == ["--model", "cursor-grok-4.6-high", "acp"]
    assert captured["base_url"] == "acp://cursor"

    captured.clear()
    alias_client, alias_model = resolve_provider_client(
        "cursor-agent", "cursor-grok-4.6-high"
    )
    assert alias_model == "cursor-grok-4.6-high"
    assert isinstance(alias_client, _FakeClient)
    assert captured["command"] == "/opt/cursor/agent"
    assert captured["args"] == ["--model", "cursor-grok-4.6-high", "acp"]
    assert captured["base_url"] == "acp://cursor"


def test_dashboard_cursor_status_is_secret_free(monkeypatch):
    monkeypatch.setattr(
        cursor_cli,
        "get_cursor_auth_status",
        lambda **_k: {
            "logged_in": True,
            "resolved_command": "/opt/cursor/agent",
            "email": "private@example.test",
            "accessToken": "do-not-return-this",
        },
    )
    from hermes_cli.web_server import _OAUTH_PROVIDER_CATALOG, _cursor_status

    status = _cursor_status()
    blob = json.dumps(status)
    assert status["logged_in"] is True
    assert status["command"] == "/opt/cursor/agent"
    assert status["token_preview"] is None
    assert "private@example.test" not in blob
    assert "do-not-return-this" not in blob
    cursor_cards = [entry for entry in _OAUTH_PROVIDER_CATALOG if entry["id"] == "cursor"]
    assert len(cursor_cards) == 1
    assert cursor_cards[0]["cli_command"] == "hermes auth add cursor"


def test_auth_add_rejects_api_key_without_reading_secret(capsys):
    args = SimpleNamespace(provider="cursor", auth_type="api_key", api_key="TOP_SECRET")
    with pytest.raises(SystemExit) as exc:
        auth_add_command(args)
    combined = str(exc.value) + capsys.readouterr().out + capsys.readouterr().err
    assert "not supported" in str(exc.value)
    assert "TOP_SECRET" not in combined


def test_model_flow_cursor_persists_exact_provider_and_model(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(
        "hermes_cli.cursor_cli.get_cursor_auth_status",
        lambda **_k: {
            "logged_in": True,
            "resolved_command": "/opt/cursor/agent",
            "command": "agent",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.cursor_cli.discover_cursor_models",
        lambda **_k: ["cursor-grok-4.6-high", "cursor-grok-4.6-xhigh"],
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *_a, **_k: "cursor-grok-4.6-high",
    )
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda model: saved.update(choice=model))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg=cfg))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows.clear_model_endpoint_credentials",
        lambda model, **_k: None,
    )
    from hermes_cli.model_setup_flows import _model_flow_cursor

    _model_flow_cursor({}, current_model="cursor-grok-4.6-high")
    assert saved["choice"] == "cursor-grok-4.6-high"
    assert saved["cfg"]["model"]["provider"] == "cursor"
    assert saved["cfg"]["model"]["base_url"] == "acp://cursor"
    assert saved["cfg"]["model"]["api_mode"] == "chat_completions"


def test_auth_commands_dispatch_cursor_aliases(monkeypatch, capsys):
    login_calls = []
    logout_calls = []
    monkeypatch.setattr(
        cursor_cli,
        "get_cursor_auth_status",
        lambda **_k: {"logged_in": False, "provider": "cursor"},
    )
    monkeypatch.setattr(
        cursor_cli,
        "login_cursor",
        lambda **kwargs: login_calls.append(kwargs) or {"logged_in": True, "command": "agent"},
    )
    auth_add_command(
        SimpleNamespace(provider="cursor-cli", auth_type="oauth", no_browser=True, timeout=30)
    )
    assert login_calls and login_calls[0]["no_browser"] is True
    assert "Cursor Agent login verified." in capsys.readouterr().out

    monkeypatch.setattr(
        cursor_cli,
        "get_cursor_auth_status",
        lambda **_k: {
            "logged_in": True,
            "provider": "cursor",
            "auth_type": "cursor_cli",
            "command": "agent",
        },
    )
    auth_status_command(SimpleNamespace(provider="cursor-agent"))
    status_out = capsys.readouterr().out
    assert "cursor: logged in" in status_out
    assert "command: agent" in status_out
    assert "token" not in status_out.lower()

    monkeypatch.setattr(
        cursor_cli,
        "logout_cursor",
        lambda **kwargs: logout_calls.append(kwargs) or {"logged_in": False},
    )
    auth_logout_command(SimpleNamespace(provider="cursor-agent"))
    assert logout_calls
    assert "Logged out of Cursor Agent." in capsys.readouterr().out


def _run_cursor_doctor(
    monkeypatch,
    tmp_path,
    *,
    logged_in: bool,
    provider_value: str = "cursor",
    default_model: str = "cursor-grok-4.6-high",
) -> str:
    import contextlib
    import io
    import sys
    import types
    from argparse import Namespace

    from hermes_cli import doctor as doctor_mod

    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        f"  provider: {provider_value}\n"
        f"  default: {default_model}\n"
        "  base_url: acp://cursor\n"
        "  api_mode: chat_completions\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.get_auth_status",
        lambda provider: (
            {
                "logged_in": logged_in,
                "configured": logged_in,
                "provider": "cursor",
                "error": None if logged_in else "Cursor Agent is not authenticated.",
            }
            if provider == "cursor"
            else {"logged_in": False}
        ),
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False, live=False))
    return buf.getvalue()


def test_doctor_flags_unauthenticated_cursor(monkeypatch, tmp_path):
    out = _run_cursor_doctor(monkeypatch, tmp_path, logged_in=False)
    assert "model.provider 'cursor' is set but Cursor Agent is not authenticated" in out
    assert "hermes auth add cursor" in out
    assert "no API key is configured" not in out


def test_doctor_flags_unauthenticated_cursor_alias(monkeypatch, tmp_path):
    out = _run_cursor_doctor(
        monkeypatch, tmp_path, logged_in=False, provider_value="cursor-agent"
    )
    assert "model.provider 'cursor' is set but Cursor Agent is not authenticated" in out
    assert "no API key is configured" not in out


def test_doctor_accepts_authenticated_cursor(monkeypatch, tmp_path):
    out = _run_cursor_doctor(monkeypatch, tmp_path, logged_in=True)
    assert "model.provider 'cursor' is set but Cursor Agent is not authenticated" not in out
    assert "no API key is configured" not in out


def test_doctor_accepts_slash_bearing_cursor_model_ids(monkeypatch, tmp_path):
    out = _run_cursor_doctor(
        monkeypatch,
        tmp_path,
        logged_in=True,
        default_model="x-ai/cursor-grok-4.6-high",
    )
    assert "uses a vendor/model slug" not in out
    assert "vendor-prefixed" not in out


def test_oneshot_forwards_cursor_command_and_args(monkeypatch):
    from hermes_cli.oneshot import _run_agent

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", mod("hermes_state", SessionDB=lambda: object()))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "cursor-grok-4.6-high"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_k: {
                "api_key": "cursor-agent",
                "base_url": "acp://cursor",
                "provider": "cursor",
                "requested_provider": "cursor",
                "api_mode": "chat_completions",
                "command": "/opt/cursor/agent",
                "args": ["--model", "cursor-grok-4.6-high", "acp"],
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_a, **_k: set()),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        mod(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_k: None,
        ),
    )

    text, result = _run_agent("ping", model="cursor-grok-4.6-high", provider="cursor")
    assert text == "ok"
    assert not result.get("failed")
    assert captured["provider"] == "cursor"
    assert captured["acp_command"] == "/opt/cursor/agent"
    assert captured["acp_args"] == ["--model", "cursor-grok-4.6-high", "acp"]
    assert captured["prompt"] == "ping"


def test_doctor_accepts_slash_bearing_ids_on_cursor_alias(monkeypatch, tmp_path):
    out = _run_cursor_doctor(
        monkeypatch,
        tmp_path,
        logged_in=True,
        provider_value="cursor-agent",
        default_model="x-ai/cursor-grok-4.6-high",
    )
    assert "uses a vendor/model slug" not in out
    assert "vendor-prefixed" not in out


def test_switch_model_forwards_exact_cursor_acp_args():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as mock_openai,
        patch("agent.copilot_acp_client.CopilotACPClient") as mock_acp_client,
    ):
        mock_acp_client.return_value = MagicMock()
        agent = AIAgent(
            api_key="cursor-agent",
            base_url="acp://cursor",
            provider="cursor",
            api_mode="chat_completions",
            model="cursor-grok-4.6-high",
            acp_command="/opt/cursor/agent",
            acp_args=["--model", "cursor-grok-4.6-high", "acp"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        mock_acp_client.reset_mock()
        agent.switch_model(
            new_model="cursor-grok-4.6-xhigh",
            new_provider="cursor",
            api_key="cursor-agent",
            base_url="acp://cursor",
            api_mode="chat_completions",
            command="/opt/cursor/agent",
            args=["--model", "cursor-grok-4.6-high", "acp"],
        )

    mock_openai.assert_not_called()
    assert mock_acp_client.call_args.kwargs["command"] == "/opt/cursor/agent"
    assert mock_acp_client.call_args.kwargs["args"] == [
        "--model",
        "cursor-grok-4.6-xhigh",
        "acp",
    ]
    assert agent.acp_command == "/opt/cursor/agent"
    assert agent.acp_args == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert agent.model == "cursor-grok-4.6-xhigh"


def test_switch_model_to_cursor_without_command_rolls_back():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.copilot_acp_client.CopilotACPClient") as mock_acp_client,
    ):
        first = MagicMock(name="original-cursor-client")
        mock_acp_client.return_value = first
        agent = AIAgent(
            api_key="cursor-agent",
            base_url="acp://cursor",
            provider="cursor",
            api_mode="chat_completions",
            model="cursor-grok-4.6-high",
            acp_command="/opt/cursor/agent",
            acp_args=["--model", "cursor-grok-4.6-high", "acp"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        original_client = agent.client
        mock_acp_client.side_effect = ValueError("Cursor ACP requires an explicit command")
        with pytest.raises(ValueError, match="explicit command"):
            agent.switch_model(
                new_model="cursor-grok-4.6-xhigh",
                new_provider="cursor",
                api_key="cursor-agent",
                base_url="acp://cursor",
                api_mode="chat_completions",
            )
        assert agent.model == "cursor-grok-4.6-high"
        assert agent.acp_args == ["--model", "cursor-grok-4.6-high", "acp"]
        assert agent.client is original_client


def test_model_switch_result_carries_cursor_command_and_args(monkeypatch):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "cursor-agent",
            "base_url": "acp://cursor",
            "provider": "cursor",
            "api_mode": "chat_completions",
            "command": "/opt/cursor/agent",
            "args": ["--model", "stale-from-resolver", "acp"],
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": "",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_info",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch._check_hermes_model_warning",
        lambda *_a, **_k: None,
    )
    result = switch_model(
        raw_input="cursor-grok-4.6-xhigh",
        current_provider="cursor",
        current_model="cursor-grok-4.6-high",
        current_base_url="acp://cursor",
        current_api_key="cursor-agent",
        explicit_provider="cursor",
    )
    assert result.success is True
    assert result.new_model == "cursor-grok-4.6-xhigh"
    assert result.command == "/opt/cursor/agent"
    assert result.args == ["--model", "cursor-grok-4.6-xhigh", "acp"]


def test_gateway_session_override_copies_cursor_command_and_args():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    sk = "telegram:1"
    override = {
        "model": "cursor-grok-4.6-xhigh",
        "provider": "cursor",
        "api_key": "cursor-agent",
        "base_url": "acp://cursor",
        "api_mode": "chat_completions",
        "command": "/opt/cursor/agent",
        "args": ["--model", "cursor-grok-4.6-xhigh", "acp"],
    }
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
    runner._peek_session_state = lambda key: state if key == sk else None
    model, rt = runner._apply_session_model_override(
        sk,
        "cursor-grok-4.6-high",
        {
            "provider": "cursor",
            "api_key": "cursor-agent",
            "base_url": "acp://cursor",
            "api_mode": "chat_completions",
            "command": "/opt/cursor/agent",
            "args": ["--model", "cursor-grok-4.6-high", "acp"],
            "credential_pool": "unchanged",
        },
    )
    assert model == "cursor-grok-4.6-xhigh"
    assert rt["args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert rt["command"] == "/opt/cursor/agent"


def test_apply_cursor_runtime_model_rewrites_stale_args():
    stale = {
        "provider": "cursor",
        "base_url": "acp://cursor",
        "command": "/opt/cursor/agent",
        "args": ["--model", "cursor-grok-4.6-high", "acp"],
        "api_key": "cursor-agent",
    }
    synced = cursor_cli.apply_cursor_runtime_model(stale, "cursor-grok-4.6-xhigh")
    assert synced["args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert stale["args"] == ["--model", "cursor-grok-4.6-high", "acp"]
    alias = cursor_cli.apply_cursor_runtime_model(
        {**stale, "provider": "cursor-agent"},
        "x-ai/cursor-grok-4.6-high",
    )
    assert alias["args"] == ["--model", "x-ai/cursor-grok-4.6-high", "acp"]
    by_url = cursor_cli.apply_cursor_runtime_model(
        {**stale, "provider": "custom"},
        "cursor-grok-4.6-xhigh",
    )
    assert by_url["args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]


def test_apply_cursor_runtime_model_leaves_copilot_and_xai_alone():
    copilot = {
        "provider": "copilot-acp",
        "base_url": "acp://copilot",
        "command": "/opt/copilot",
        "args": ["--acp", "--stdio"],
    }
    assert cursor_cli.apply_cursor_runtime_model(copilot, "gpt-5.4")["args"] == [
        "--acp",
        "--stdio",
    ]
    xai = {
        "provider": "xai",
        "base_url": "https://api.x.ai/v1",
        "command": None,
        "args": [],
    }
    assert cursor_cli.apply_cursor_runtime_model(xai, "grok-4")["args"] == []


def test_gateway_turn_config_refreshes_stale_cursor_args():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._service_tier = None
    route = runner._resolve_turn_agent_config(
        "hello",
        "cursor-grok-4.6-xhigh",
        {
            "provider": "cursor",
            "api_key": "cursor-agent",
            "base_url": "acp://cursor",
            "api_mode": "chat_completions",
            "command": "/opt/cursor/agent",
            "args": ["--model", "cursor-grok-4.6-high", "acp"],
            "credential_pool": None,
            "max_tokens": None,
        },
    )
    assert route["model"] == "cursor-grok-4.6-xhigh"
    assert route["runtime"]["args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert route["signature"][6] == ("--model", "cursor-grok-4.6-xhigh", "acp")


def test_is_cursor_runtime_accepts_aliases_and_rejects_neighbors():
    assert cursor_cli.is_cursor_runtime(provider="cursor")
    assert cursor_cli.is_cursor_runtime(provider="cursor-agent")
    assert cursor_cli.is_cursor_runtime(provider="cursor-cli", base_url="")
    assert cursor_cli.is_cursor_runtime(provider="custom", base_url="acp://cursor/")
    assert not cursor_cli.is_cursor_runtime(provider="copilot-acp", base_url="acp://copilot")
    assert not cursor_cli.is_cursor_runtime(provider="custom", base_url="acp://cursor-evil")
    assert not cursor_cli.is_cursor_runtime(provider="xai", base_url="https://api.x.ai/v1")


def test_cli_turn_config_refreshes_stale_cursor_args():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    cli = CLIAgentSetupMixin()
    cli.api_key = "cursor-agent"
    cli.base_url = "acp://cursor"
    cli.provider = "cursor"
    cli.requested_provider = "cursor"
    cli.api_mode = "chat_completions"
    cli.acp_command = "/opt/cursor/agent"
    cli.acp_args = ["--model", "cursor-grok-4.6-high", "acp"]
    cli.model = "cursor-grok-4.6-xhigh"
    cli._credential_pool = None
    cli.service_tier = None

    route = cli._resolve_turn_agent_config("hello")
    assert route["model"] == "cursor-grok-4.6-xhigh"
    assert route["runtime"]["args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert route["signature"][6] == ("--model", "cursor-grok-4.6-xhigh", "acp")


def test_cli_ensure_runtime_syncs_cursor_args_after_model_default(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    fake_cli = types.ModuleType("cli")
    fake_cli.ChatConsole = lambda: SimpleNamespace(print=lambda *a, **k: None)
    fake_cli._cprint = lambda *a, **k: None
    fake_cli.logger = SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    class _CLI(CLIAgentSetupMixin):
        def _normalize_model_for_provider(self, _provider):
            return False

    cli = _CLI()
    cli.requested_provider = "cursor"
    cli._explicit_api_key = None
    cli._explicit_base_url = None
    cli.model = ""
    cli.api_key = None
    cli.base_url = None
    cli.provider = None
    cli.api_mode = "chat_completions"
    cli.acp_command = None
    cli.acp_args = []
    cli._credential_pool = None
    cli.agent = None
    cli._fallback_model = []

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "cursor-agent",
            "base_url": "acp://cursor",
            "provider": "cursor",
            "api_mode": "chat_completions",
            "command": "/opt/cursor/agent",
            "args": ["--model", "stale-from-config", "acp"],
            "source": "cursor-cli",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.get_default_model_for_provider",
        lambda _provider: "cursor-grok-4.6-high",
    )

    assert cli._ensure_runtime_credentials() is True
    assert cli.model == "cursor-grok-4.6-high"
    assert cli.acp_command == "/opt/cursor/agent"
    assert cli.acp_args == ["--model", "cursor-grok-4.6-high", "acp"]


def _cursor_parent():
    parent = MagicMock()
    parent.base_url = "acp://cursor"
    parent.api_key = "cursor-agent"
    parent.provider = "cursor"
    parent.api_mode = "chat_completions"
    parent.model = "cursor-grok-4.6-high"
    parent.acp_command = "/opt/cursor/agent"
    parent.acp_args = ["--model", "cursor-grok-4.6-high", "acp"]
    parent.platform = "cli"
    parent.enabled_toolsets = ["terminal"]
    parent.disabled_toolsets = []
    parent.valid_tool_names = ["terminal"]
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.max_tokens = None
    parent.prefill_messages = None
    parent._fallback_chain = []
    parent.request_overrides = {}
    parent.session_id = "parent"
    parent.client = None
    parent._client_kwargs = {
        "base_url": "acp://cursor",
        "command": "/opt/cursor/agent",
        "args": ["--model", "cursor-grok-4.6-high", "acp"],
    }
    return parent


def test_delegate_child_keeps_cursor_identity_and_exact_args(monkeypatch):
    from tools.delegate_tool import _build_child_agent

    monkeypatch.setattr("shutil.which", lambda command: command)
    parent = _cursor_parent()
    with patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="keep cursor identity",
            context=None,
            toolsets=None,
            model="cursor-grok-4.6-high",
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            override_provider="cursor",
            override_base_url="acp://cursor",
            override_api_key="cursor-agent",
            override_api_mode="chat_completions",
            override_acp_command="/opt/cursor/agent",
            override_acp_args=["--model", "cursor-grok-4.6-high", "acp"],
        )
    kwargs = mock_agent.call_args.kwargs
    assert kwargs["provider"] == "cursor"
    assert kwargs["base_url"] == "acp://cursor"
    assert kwargs["acp_command"] == "/opt/cursor/agent"
    assert kwargs["acp_args"] == ["--model", "cursor-grok-4.6-high", "acp"]


def test_delegate_child_rewrites_stale_inherited_cursor_args():
    from tools.delegate_tool import _build_child_agent

    parent = _cursor_parent()
    with patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="use a different cursor model",
            context=None,
            toolsets=None,
            model="cursor-grok-4.6-xhigh",
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
        )
    kwargs = mock_agent.call_args.kwargs
    assert kwargs["provider"] == "cursor"
    assert kwargs["acp_args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]


def test_delegate_acp_override_still_pins_copilot_acp(monkeypatch):
    from tools.delegate_tool import _build_child_agent

    monkeypatch.setattr("shutil.which", lambda command: command)
    parent = _cursor_parent()
    parent.provider = "openrouter"
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.acp_command = None
    parent.acp_args = []
    with patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="legacy copilot acp override",
            context=None,
            toolsets=None,
            model="gpt-5.4",
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            override_provider="openrouter",
            override_base_url="acp://copilot",
            override_api_key="copilot-acp",
            override_acp_command="/opt/copilot",
            override_acp_args=["--acp", "--stdio"],
        )
    kwargs = mock_agent.call_args.kwargs
    assert kwargs["provider"] == "copilot-acp"
    assert kwargs["acp_args"] == ["--acp", "--stdio"]


def test_aiagent_rewrites_stale_cursor_args_and_pins_marker_url():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as mock_openai,
        patch("agent.copilot_acp_client.CopilotACPClient") as mock_acp_client,
    ):
        mock_acp_client.return_value = MagicMock()
        agent = AIAgent(
            api_key="cursor-agent",
            provider="cursor",
            api_mode="chat_completions",
            model="cursor-grok-4.6-xhigh",
            acp_command="/opt/cursor/agent",
            acp_args=["--model", "cursor-grok-4.6-high", "acp"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    mock_openai.assert_not_called()
    assert agent.base_url == "acp://cursor"
    assert agent.acp_args == ["--model", "cursor-grok-4.6-xhigh", "acp"]
    assert mock_acp_client.call_args.kwargs["base_url"] == "acp://cursor"
    assert mock_acp_client.call_args.kwargs["args"] == [
        "--model",
        "cursor-grok-4.6-xhigh",
        "acp",
    ]


def test_tui_background_kwargs_rewrites_stale_cursor_args(monkeypatch):
    from tui_gateway import server

    agent = SimpleNamespace(
        base_url="acp://cursor",
        api_key="cursor-agent",
        provider="cursor",
        api_mode="chat_completions",
        acp_command="/opt/cursor/agent",
        acp_args=["--model", "cursor-grok-4.6-high", "acp"],
        model="cursor-grok-4.6-xhigh",
        enabled_toolsets=["terminal"],
        ephemeral_system_prompt=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        reasoning_config=None,
        service_tier=None,
        request_overrides={},
        _fallback_chain=None,
        _fallback_model=None,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["terminal"])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_reasoning_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    kwargs = server._background_agent_kwargs(agent, "task-id")
    assert kwargs["model"] == "cursor-grok-4.6-xhigh"
    assert kwargs["acp_command"] == "/opt/cursor/agent"
    assert kwargs["acp_args"] == ["--model", "cursor-grok-4.6-xhigh", "acp"]
