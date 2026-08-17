"""First-class Cursor Agent CLI integration for Hermes.

Cursor owns subscription/browser credentials and their refresh lifecycle. Hermes
only launches the supported CLI commands, consumes secret-free status/model
metadata, and uses Cursor's ACP stdio transport for inference.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any, Mapping

from agent.redact import redact_sensitive_text
from hermes_constants import display_hermes_home, get_real_home
from tools.environments.local import hermes_subprocess_env

CURSOR_MARKER_BASE_URL = "acp://cursor"
DEFAULT_CURSOR_COMMAND = "agent"
_INVALID_COMMAND_CHARS = frozenset("\x00\n\r;|&$`<>")
_COPILOT_ACP_ENV_KEYS = (
    "HERMES_COPILOT_ACP_COMMAND",
    "HERMES_COPILOT_ACP_ARGS",
    "COPILOT_CLI_PATH",
    "COPILOT_ACP_BASE_URL",
)
DEFAULT_CURSOR_MODELS: tuple[str, ...] = (
    "cursor-grok-4.6-high",
    "cursor-grok-4.6-xhigh",
    "cursor-grok-4.6-high-fast",
    "cursor-grok-4.6-xhigh-fast",
    "cursor-grok-4.6-medium",
    "cursor-grok-4.6-low",
)

# `agent models` currently emits `<id> - <display name>`. Keep parsing strict:
# no whitespace/control characters, shell metacharacters, or option-like IDs.
_MODEL_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._:+/\[\]=,-]{0,191})\s+-\s+\S.*$"
)


class CursorCLIError(RuntimeError):
    """Typed, secret-safe Cursor CLI failure."""

    def __init__(self, message: str, *, code: str = "cursor_cli_error") -> None:
        super().__init__(message)
        self.code = code


def _configured_cursor_command(config: Mapping[str, Any] | None = None) -> str:
    if config is None:
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
            config = loaded if isinstance(loaded, Mapping) else {}
        except Exception:
            config = {}

    blocks: list[Any] = []
    if isinstance(config, Mapping):
        blocks.append(config.get("cursor"))
        providers = config.get("providers")
        if isinstance(providers, Mapping):
            blocks.append(providers.get("cursor"))
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        value = block.get("command") or block.get("cli_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_CURSOR_COMMAND


def _validate_cursor_command(command: str) -> str:
    value = str(command or "").strip()
    if (
        not value
        or value.startswith("-")
        or any(ch in value for ch in _INVALID_COMMAND_CHARS)
    ):
        raise CursorCLIError(
            "Cursor command must be a single executable path, not a shell "
            "command or argument string.",
            code="invalid_cursor_command",
        )
    return value


def _resolve_named_command(
    command: str,
    *,
    require: bool = True,
) -> tuple[str, str | None]:
    command = _validate_cursor_command(command)
    resolved = shutil.which(command)
    if require and not resolved:
        raise CursorCLIError(
            f"Could not find Cursor Agent CLI command '{command}'. Install Cursor "
            f"Agent or set cursor.command in {display_hermes_home()}/config.yaml.",
            code="missing_cursor_cli",
        )
    return command, resolved


def resolve_cursor_command(
    config: Mapping[str, Any] | None = None,
    *,
    require: bool = True,
) -> tuple[str, str | None]:
    """Return the configured command and its executable path.

    The command is a single executable path/name, never a shell fragment or an
    argv string. Arguments are owned by Hermes and assembled separately.
    """

    return _resolve_named_command(
        _configured_cursor_command(config),
        require=require,
    )


def _cursor_child_env(*, no_browser: bool = False) -> dict[str, str]:
    # Cursor is the model-driving process, but its supported browser login owns
    # credentials in Cursor's store. Do not forward unrelated Hermes provider or
    # gateway credentials merely to run status/login/model discovery.
    env = hermes_subprocess_env(inherit_credentials=False)
    # Match ACP runtime: Cursor credentials live in the OS user home, not a
    # Hermes profile HOME that hermes_subprocess_env may have applied.
    real_home = get_real_home(env)
    if real_home:
        env["HOME"] = real_home
    for key in _COPILOT_ACP_ENV_KEYS:
        env.pop(key, None)
    if no_browser:
        env["NO_OPEN_BROWSER"] = "1"
    else:
        env.pop("NO_OPEN_BROWSER", None)
    return env


def _safe_process_detail(stderr: str) -> str:
    text = redact_sensitive_text(str(stderr or ""), force=True).strip()
    if not text:
        return ""
    line = text.splitlines()[-1].strip()
    return line[:240]


def get_cursor_auth_status(
    *,
    command: str | None = None,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Return a secret-free live authentication status from Cursor Agent."""

    configured = command or _configured_cursor_command()
    try:
        if command:
            configured, resolved = _resolve_named_command(command, require=True)
        else:
            configured, resolved = resolve_cursor_command(require=True)
        assert resolved is not None
        completed = subprocess.run(
            [resolved, "status", "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout_seconds)),
            env=_cursor_child_env(),
        )
    except CursorCLIError as exc:
        return {
            "provider": "cursor",
            "configured": False,
            "logged_in": False,
            "command": configured,
            "error": str(exc),
            "error_code": exc.code,
        }
    except subprocess.TimeoutExpired:
        return {
            "provider": "cursor",
            "configured": True,
            "logged_in": False,
            "command": configured,
            "error": "Cursor authentication status timed out.",
            "error_code": "cursor_status_timeout",
        }
    except OSError as exc:
        return {
            "provider": "cursor",
            "configured": False,
            "logged_in": False,
            "command": configured,
            "error": f"Could not start Cursor Agent CLI: {exc.__class__.__name__}.",
            "error_code": "cursor_status_start_failed",
        }

    if completed.returncode != 0:
        detail = _safe_process_detail(completed.stderr)
        message = "Cursor authentication status failed."
        if detail:
            message = f"{message} {detail}"
        return {
            "provider": "cursor",
            "configured": True,
            "logged_in": False,
            "command": configured,
            "resolved_command": resolved,
            "error": message,
            "error_code": "cursor_status_failed",
        }

    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return {
            "provider": "cursor",
            "configured": True,
            "logged_in": False,
            "command": configured,
            "resolved_command": resolved,
            "error": "Cursor returned malformed authentication status JSON.",
            "error_code": "cursor_status_malformed",
        }

    authenticated = payload.get("isAuthenticated") is True or str(
        payload.get("status") or ""
    ).strip().lower() == "authenticated"
    result: dict[str, Any] = {
        "provider": "cursor",
        "configured": True,
        "logged_in": authenticated,
        "auth_type": "cursor_cli",
        "command": configured,
        "resolved_command": resolved,
    }
    if not authenticated:
        result["error"] = "Cursor Agent is not authenticated. Run `hermes auth add cursor`."
        result["error_code"] = "cursor_not_authenticated"
    return result


def login_cursor(
    *,
    no_browser: bool = False,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run Cursor's supported login flow and verify the resulting status."""

    command, resolved = resolve_cursor_command(require=True)
    assert resolved is not None
    before = get_cursor_auth_status(command=resolved)
    if before.get("logged_in"):
        return before
    try:
        completed = subprocess.run(
            [resolved, "login"],
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
            env=_cursor_child_env(no_browser=no_browser),
        )
    except subprocess.TimeoutExpired as exc:
        raise CursorCLIError(
            "Cursor login timed out before authentication completed.",
            code="cursor_login_timeout",
        ) from exc
    except OSError as exc:
        raise CursorCLIError(
            f"Could not start Cursor login: {exc.__class__.__name__}.",
            code="cursor_login_start_failed",
        ) from exc
    if completed.returncode != 0:
        raise CursorCLIError(
            f"Cursor login exited with status {completed.returncode}.",
            code="cursor_login_failed",
        )
    status = get_cursor_auth_status(command=resolved)
    if not status.get("logged_in"):
        raise CursorCLIError(
            status.get("error") or "Cursor login completed but authentication was not established.",
            code=str(status.get("error_code") or "cursor_login_unverified"),
        )
    status["command"] = command
    return status


def logout_cursor(*, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Ask Cursor to clear its credentials and verify the logged-out state."""

    command, resolved = resolve_cursor_command(require=True)
    assert resolved is not None
    try:
        completed = subprocess.run(
            [resolved, "logout"],
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
            env=_cursor_child_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise CursorCLIError(
            "Cursor logout timed out.", code="cursor_logout_timeout"
        ) from exc
    except OSError as exc:
        raise CursorCLIError(
            f"Could not start Cursor logout: {exc.__class__.__name__}.",
            code="cursor_logout_start_failed",
        ) from exc
    if completed.returncode != 0:
        raise CursorCLIError(
            f"Cursor logout exited with status {completed.returncode}.",
            code="cursor_logout_failed",
        )
    status = get_cursor_auth_status(command=resolved)
    if status.get("logged_in"):
        raise CursorCLIError(
            "Cursor logout completed but the CLI still reports an authenticated session.",
            code="cursor_logout_unverified",
        )
    status["command"] = command
    return status


def parse_cursor_models(output: str) -> list[str]:
    """Parse exact model IDs from the stable human-readable Cursor listing."""

    result: list[str] = []
    seen: set[str] = set()
    for line in str(output or "").splitlines():
        match = _MODEL_LINE_RE.match(line)
        if not match:
            continue
        model_id = match.group(1)
        key = model_id.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(model_id)
    return result


def discover_cursor_models(
    *,
    command: str | None = None,
    timeout_seconds: float = 30.0,
) -> list[str]:
    """Return the authenticated account's exact Cursor model IDs."""

    _configured, resolved = (
        _resolve_named_command(command, require=True)
        if command
        else resolve_cursor_command(require=True)
    )
    if not resolved:
        raise CursorCLIError(
            f"Could not find Cursor Agent CLI command '{command or _configured}'.",
            code="missing_cursor_cli",
        )
    try:
        completed = subprocess.run(
            [resolved, "models"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout_seconds)),
            env=_cursor_child_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise CursorCLIError(
            "Cursor model discovery timed out.", code="cursor_models_timeout"
        ) from exc
    except OSError as exc:
        raise CursorCLIError(
            f"Could not start Cursor model discovery: {exc.__class__.__name__}.",
            code="cursor_models_start_failed",
        ) from exc
    if completed.returncode != 0:
        detail = _safe_process_detail(completed.stderr)
        message = "Cursor model discovery failed."
        if detail:
            message = f"{message} {detail}"
        raise CursorCLIError(message, code="cursor_models_failed")
    models = parse_cursor_models(completed.stdout)
    if not models:
        raise CursorCLIError(
            "Cursor model discovery returned no valid model IDs.",
            code="cursor_models_empty",
        )
    return models


def cursor_acp_args(model: str) -> list[str]:
    """Build Cursor's ACP argv without passing any credential in process args."""

    selected = str(model or "").strip()
    if not selected or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:+/\[\]=,-]{0,191}", selected):
        raise CursorCLIError(
            "Cursor requires a valid selected model before starting ACP.",
            code="invalid_cursor_model",
        )
    return ["--model", selected, "acp"]


def is_cursor_runtime(*, provider: str = "", base_url: str = "") -> bool:
    """True when provider identity or ACP marker URL is first-class Cursor."""
    name = str(provider or "").strip().lower()
    try:
        from hermes_cli.models import normalize_provider

        name = normalize_provider(name)
    except Exception:
        pass
    url = str(base_url or "").strip().rstrip("/").lower()
    return name == "cursor" or url == CURSOR_MARKER_BASE_URL or url.startswith(
        CURSOR_MARKER_BASE_URL + "/"
    )


def apply_cursor_runtime_model(
    runtime: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    """Rewrite Cursor ACP argv so a later model override cannot keep a stale ``--model``.

    Copilot ACP args are not model-specific and are left unchanged.
    """
    result = dict(runtime)
    if not is_cursor_runtime(
        provider=str(result.get("provider") or ""),
        base_url=str(result.get("base_url") or ""),
    ):
        return result
    selected = str(model or "").strip()
    if not selected:
        return result
    result["args"] = cursor_acp_args(selected)
    return result
