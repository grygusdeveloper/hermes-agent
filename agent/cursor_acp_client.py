"""Cursor Agent ACP transport. Cursor owns authentication; Hermes selects the model at spawn."""
from __future__ import annotations

import contextvars
import re
from typing import Any

from agent.copilot_acp_client import CopilotACPClient
from hermes_constants import get_real_home
from tools.environments.local import hermes_subprocess_env

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/\[\]=,-]{0,191}\Z")
_COPILOT_KEYS = ("HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH", "COPILOT_ACP_BASE_URL")
_selected_model: contextvars.ContextVar[str] = contextvars.ContextVar("cursor_acp_model", default="")


class CursorACPClient(CopilotACPClient):
    """Reuse the ACP wire/FS safety bridge, not Copilot's model selection or credential env."""

    _expected_base_url = "acp://cursor"

    def __init__(self, *, command: str | None = None, acp_command: str | None = None,
                 args: list[str] | None = None, acp_args: list[str] | None = None, **kwargs: Any):
        # Never fall through to Copilot command/args environment overrides.
        super().__init__(command=command or acp_command or "agent", args=args or acp_args or ["acp"], **kwargs)

    def _process_args(self) -> list[str]:
        selected = _selected_model.get()
        return ["--model", selected, "acp"] if selected else ["acp"]

    def _process_env(self) -> dict[str, str]:
        env = hermes_subprocess_env(inherit_credentials=False)
        if home := get_real_home(env):
            env["HOME"] = home
        for key in _COPILOT_KEYS:
            env.pop(key, None)
        return env

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float,
                    model: str | None = None) -> tuple[str, str]:
        selected = str(model or "").strip()
        if not _MODEL_ID.fullmatch(selected):
            raise ValueError("Cursor ACP requires a valid model ID before starting the CLI.")
        token = _selected_model.set(selected)
        try:
            with self._session(timeout_seconds) as (session, request):
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                request("session/prompt", {
                    "sessionId": session["sessionId"],
                    "prompt": [{"type": "text", "text": prompt_text}],
                }, text_parts=text_parts, reasoning_parts=reasoning_parts)
                text = "".join(text_parts)
                # Cursor returns a successful ACP end_turn with this text when the
                # account is exhausted. Do not publish it as the model's answer.
                if text.strip() == "Upgrade your plan to continue":
                    raise RuntimeError("Cursor account usage limit reached: upgrade your plan to continue")
                return text, "".join(reasoning_parts)
        finally:
            _selected_model.reset(token)
