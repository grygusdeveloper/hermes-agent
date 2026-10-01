"""Reasoning-aware ``model_aliases``: an alias may carry ``reasoning_effort`` and picking the alias
itself (never a full model ID that several aliases share) reports that preset on the switch result,
which the gateway applies as session-scoped reasoning."""
from unittest.mock import patch

import pytest

import hermes_cli.model_switch as ms

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}
_CODEX = "https://chatgpt.com/backend-api/codex"


def _load(monkeypatch, aliases: dict) -> dict:
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model_aliases": aliases})
    return ms._load_direct_aliases()


def test_alias_loads_validated_reasoning_effort(monkeypatch):
    aliases = _load(monkeypatch, {
        "sol-xhigh": {"model": "gpt-5.6-sol", "provider": "openai-codex", "base_url": _CODEX,
                      "reasoning_effort": "XHigh"},
        "terse": {"model": "gpt-5.6-luna", "provider": "openai-codex", "reasoning": "low"},
        "typo": {"model": "gpt-5.6-sol", "provider": "openai-codex", "reasoning_effort": "xhgh"},
        "plain": {"model": "gpt-5.6-sol", "provider": "openai-codex"},
    })
    assert aliases["sol-xhigh"].reasoning_effort == "xhigh"
    assert aliases["terse"].reasoning_effort == "low"
    # An unknown level is dropped rather than applied or advertised.
    assert aliases["typo"].reasoning_effort == ""
    assert aliases["plain"].reasoning_effort == ""


def _switch(monkeypatch, raw_input: str, aliases: dict):
    monkeypatch.setattr(ms, "DIRECT_ALIASES", aliases)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "test-key", "base_url": _CODEX, "api_mode": "codex_responses"}), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
        return ms.switch_model(raw_input=raw_input, current_provider="openai-codex",
                               current_model="gpt-5.6-sol", current_base_url=_CODEX,
                               current_api_key="test-key", user_providers={}, custom_providers=[])


@pytest.mark.parametrize("typed", ["sol-xhigh", "SOL-XHIGH"])
def test_alias_reasoning_effort_reaches_the_switch_result(monkeypatch, typed):
    result = _switch(monkeypatch, typed, {
        "sol-xhigh": ms.DirectAlias("gpt-5.6-sol", "openai-codex", _CODEX, reasoning_effort="xhigh"),
    })
    assert result.success, result.error_message
    assert (result.new_model, result.reasoning_effort) == ("gpt-5.6-sol", "xhigh")


def test_full_model_id_does_not_inherit_a_duplicate_alias_effort(monkeypatch):
    result = _switch(monkeypatch, "gpt-5.6-sol", {
        "sol-high": ms.DirectAlias("gpt-5.6-sol", "openai-codex", _CODEX, reasoning_effort="high"),
        "sol-xhigh": ms.DirectAlias("gpt-5.6-sol", "openai-codex", _CODEX, reasoning_effort="xhigh"),
    })
    assert result.success, result.error_message
    assert result.new_model == "gpt-5.6-sol"
    # The full ID reverse-resolves to one of the aliases, yet carries neither alias's preset.
    assert result.resolved_via_alias in {"sol-high", "sol-xhigh"}
    assert result.reasoning_effort == ""
