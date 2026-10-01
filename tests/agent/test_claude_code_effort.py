"""Claude Code effort and thinking reach the launched CLI (ported from WP4 7025953856 and 96005fcd37).

The end-to-end cases drive a real AIAgent through the real transport, provider profile, client and
session; only the ``claude`` subprocess is the stream-json fixture from test_claude_code_provider.
"""
import json

import pytest

from agent.claude_code_client import cli_reasoning
from hermes_constants import resolve_reasoning_config
from tests.agent.test_claude_code_provider import native, rows, scope  # noqa: F401  (fixture)


def _actor(tmp_path, **kwargs):
    from run_agent import AIAgent

    return AIAgent(provider="claude-code", model="opus", base_url="acp://claude-code", api_key="claude-code",
                   command=str(tmp_path / "claude"), args=[], session_id="effort-owner", enabled_toolsets=[],
                   skip_memory=True, skip_context_files=True, quiet_mode=True, save_trajectories=False,
                   max_iterations=2, **kwargs)


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("agent_cfg, effort, thinking", [
    # Per-model override beats the global effort (the live config's shape).
    ({"reasoning_effort": "medium", "reasoning_overrides": {"opus": "xhigh"}}, "xhigh", None),
    ({"reasoning_effort": "high"}, "high", None),
    ({"reasoning_effort": "medium", "reasoning_overrides": {"opus": "minimal"}}, "low", None),
    ({"reasoning_effort": "ultra"}, "max", None),
    ({"reasoning_effort": "max"}, "max", None),
    # Reasoning off: thinking disabled at the lowest effort.
    ({"reasoning_effort": "medium", "reasoning_overrides": {"opus": "none"}}, "low", "disabled"),
    ({}, None, None),
])
def test_configured_effort_reaches_the_launched_cli(native, tmp_path, agent_cfg, effort, thinking):
    """Regression: no reasoning reached ClaudeCodeClient, so reasoning_overrides and alias efforts
    were ignored and every Claude turn ran at the CLI default."""
    reasoning = resolve_reasoning_config({"agent": agent_cfg}, "opus")
    with scope(tmp_path / "a", "effort-owner"):
        agent = _actor(tmp_path, reasoning_config=reasoning)
        try:
            answer = agent.run_conversation(json.dumps({"scenario": "text"}))
        finally:
            agent.close()
    assert answer["final_response"] == "Native answer."
    argv = rows(tmp_path)[-1]["args"]
    assert _flag(argv, "--effort") == effort
    assert _flag(argv, "--thinking") == thinking


@pytest.mark.parametrize("requested, expected", [
    ((None, None), (None, None)),
    (("high", None), ("high", None)),
    (("xhigh", None), ("xhigh", None)),
    (("minimal", None), ("low", None)),
    (("ultra", None), ("max", None)),
    # Ultracode is xhigh plus Claude Code's own orchestration, whose native tools are off here.
    (("ultracode", None), ("xhigh", None)),
    (("none", None), ("low", "disabled")),
    (("high", "disabled"), ("low", "disabled")),
    ((" HIGH ", None), ("high", None)),
])
def test_hermes_levels_map_onto_cli_flags(requested, expected):
    assert cli_reasoning(*requested) == expected


def test_unknown_effort_or_thinking_mode_is_rejected():
    with pytest.raises(ValueError, match="reasoning effort"):
        cli_reasoning("turbo")
    with pytest.raises(ValueError, match="thinking mode"):
        cli_reasoning("high", "sometimes")


def test_profile_hands_reasoning_to_the_client_and_no_body_fields():
    from providers import get_provider_profile

    profile = get_provider_profile("claude-code")
    hook = profile.build_api_kwargs_extras
    assert hook(reasoning_config={"enabled": True, "effort": "xhigh"}) == ({}, {"reasoning_effort": "xhigh"})
    assert hook(reasoning_config={"enabled": False}) == ({}, {"reasoning_effort": "low", "thinking": "disabled"})
    assert hook(reasoning_config={"enabled": True, "effort": "none"}) == (
        {}, {"reasoning_effort": "low", "thinking": "disabled"})
    assert hook(reasoning_config={"enabled": True}) == ({}, {})
    assert hook(reasoning_config=None) == ({}, {})
