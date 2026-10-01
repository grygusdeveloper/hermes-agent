"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability,
and the PINNED_THRESHOLDS escape-hatch for read_file.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_RESULT_SIZE_CHARS,
    PINNED_THRESHOLDS,
    BudgetConfig,
    budget_for_context_window,
    budget_for_transport,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# BudgetConfig defaults
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Custom construction
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# resolve_threshold() priority chain
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    """Priority: pinned > tool_overrides > registry > default."""

    def test_pinned_wins_over_override(self):
        """Even if tool_overrides contains read_file, pinned value wins."""
        cfg = BudgetConfig(tool_overrides={"read_file": 1})
        result = cfg.resolve_threshold("read_file")
        assert result == float("inf")

    def test_tool_override_wins_over_default(self):
        """tool_overrides should be returned before falling back to registry."""
        cfg = BudgetConfig(tool_overrides={"my_tool": 42})
        result = cfg.resolve_threshold("my_tool")
        assert result == 42


    @patch("tools.registry.registry")
    def test_registry_value_capped_at_default(self, mock_registry):
        """A scaled-down budget caps an oversized registry value (#23767).

        web/terminal/x_search register max_result_size_chars=100_000; a small
        model's scaled budget must not be re-inflated by that.
        """
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig(default_result_size=30_000)
        assert cfg.resolve_threshold("web_search") == 30_000


    @patch("tools.registry.registry")
    def test_default_budget_unchanged_for_100k_tool(self, mock_registry):
        """Default budget keeps 100K registry tools at 100K (no behavior change)."""
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig()  # default_result_size == 100_000
        assert cfg.resolve_threshold("web_search") == 100_000


# ---------------------------------------------------------------------------
# budget_for_context_window() — context-aware scaling (#23767)
# ---------------------------------------------------------------------------


class TestBudgetForContextWindow:
    """Scaling the tool-output budget to the active model's context window."""

    def test_none_returns_default(self):
        assert budget_for_context_window(None) is DEFAULT_BUDGET

    def test_zero_or_negative_returns_default(self):
        assert budget_for_context_window(0) is DEFAULT_BUDGET
        assert budget_for_context_window(-5) is DEFAULT_BUDGET


    def test_scaled_budget_constrains_oversized_result(self):
        """A 279K-char result against a 65K model exceeds the scaled per-result
        threshold, so it will be persisted/truncated rather than sent whole."""
        cfg = budget_for_context_window(65_536)
        huge_len = 279_549
        threshold = cfg.resolve_threshold("mcp_firecrawl_firecrawl_search")
        assert threshold < huge_len
        assert cfg.default_result_size < huge_len


# ---------------------------------------------------------------------------
# MCP-prefix threshold (mcp_result_size)
# ---------------------------------------------------------------------------


class TestMcpPrefixThreshold:
    """mcp_* tools get the tighter 50K default, config-overridable."""

    def test_default_mcp_threshold_is_tighter_than_generic(self):
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        assert DEFAULT_MCP_RESULT_SIZE_CHARS < DEFAULT_RESULT_SIZE_CHARS
        assert DEFAULT_BUDGET.resolve_threshold("mcp_composio_search_tools") == DEFAULT_MCP_RESULT_SIZE_CHARS

    def test_non_mcp_tools_keep_generic_default(self):
        assert DEFAULT_BUDGET.resolve_threshold("some_random_tool") == DEFAULT_RESULT_SIZE_CHARS

    def test_pinned_wins_over_mcp_prefix(self):
        with patch.dict(PINNED_THRESHOLDS, {"mcp_pinned_tool": float("inf")}):
            assert DEFAULT_BUDGET.resolve_threshold("mcp_pinned_tool") == float("inf")

    def test_tool_override_wins_over_mcp_prefix(self):
        cfg = BudgetConfig(tool_overrides={"mcp_special": 75_000})
        assert cfg.resolve_threshold("mcp_special") == 75_000

    def test_mcp_threshold_capped_by_scaled_default(self):
        """On a small model the scaled default_result_size caps the MCP value."""
        cfg = BudgetConfig(default_result_size=20_000, mcp_result_size=50_000)
        assert cfg.resolve_threshold("mcp_anything") == 20_000


    def test_config_override_via_hermes_home(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_composio_multi_execute") == 30_000
        # Generic tools are untouched by the MCP knob.
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_config_override_survives_window_scaling(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(200_000)
        assert cfg.mcp_result_size == 30_000

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("tool_budget: not-a-mapping\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_x_y") == DEFAULT_MCP_RESULT_SIZE_CHARS

    def test_scaled_small_window_caps_mcp_threshold(self, tmp_path, monkeypatch):
        """A tiny model's scaled default_result_size caps even the MCP value."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
        cfg = budget_for_context_window(16_384)  # scaled default < 50K
        assert cfg.default_result_size < 50_000
        assert cfg.resolve_threshold("mcp_tool") == cfg.default_result_size


# ---------------------------------------------------------------------------
# Transport clamp (Antigravity: one argv value per request)
# ---------------------------------------------------------------------------


class TestTransportBudget:
    def test_antigravity_turn_budget_fits_one_inline_prompt_in_utf8(self):
        from agent.antigravity_session import INLINE_PROMPT_LIMIT_BYTES
        from tools.budget_config import ANTIGRAVITY_TURN_BUDGET_CHARS

        assert ANTIGRAVITY_TURN_BUDGET_CHARS * 4 < INLINE_PROMPT_LIMIT_BYTES

    @pytest.mark.parametrize(
        "provider, base_url",
        [("antigravity", "acp://antigravity"), ("antigravity", None), (None, "acp://antigravity/")],
    )
    def test_antigravity_clamps_large_context_budget(self, provider, base_url):
        from tools.budget_config import (
            ANTIGRAVITY_PREVIEW_SIZE_CHARS, ANTIGRAVITY_RESULT_SIZE_CHARS, ANTIGRAVITY_TURN_BUDGET_CHARS,
        )

        cfg = budget_for_transport(
            dataclasses.replace(budget_for_context_window(1_048_576), tool_overrides={"terminal": 80_000}),
            provider=provider, base_url=base_url,
        )
        assert cfg.turn_budget == ANTIGRAVITY_TURN_BUDGET_CHARS
        assert cfg.preview_size == ANTIGRAVITY_PREVIEW_SIZE_CHARS
        assert cfg.resolve_threshold("terminal") == ANTIGRAVITY_RESULT_SIZE_CHARS
        assert cfg.resolve_threshold("mcp_server_tool") == ANTIGRAVITY_RESULT_SIZE_CHARS
        assert cfg.resolve_threshold("web_search") <= ANTIGRAVITY_RESULT_SIZE_CHARS
        assert cfg.resolve_threshold("read_file") == float("inf")

    @pytest.mark.parametrize(
        "provider, base_url",
        [("openai-codex", "https://chatgpt.com/backend-api/codex"), ("copilot-acp", "acp://antigravity"),
         ("claude-code", "acp://claude-code"), (None, None)],
    )
    def test_other_transports_are_unchanged(self, provider, base_url):
        original = budget_for_context_window(1_048_576)
        assert budget_for_transport(original, provider=provider, base_url=base_url) is original

    def test_agent_budget_uses_the_live_transport(self):
        from agent.tool_executor import _budget_for_agent
        from tools.budget_config import ANTIGRAVITY_TURN_BUDGET_CHARS

        agent = SimpleNamespace(
            provider="antigravity",
            base_url="acp://antigravity",
            context_compressor=SimpleNamespace(context_length=1_048_576),
        )
        assert _budget_for_agent(agent).turn_budget == ANTIGRAVITY_TURN_BUDGET_CHARS
        # The client's transport counts when the agent itself carries none.
        agent = SimpleNamespace(
            context_compressor=SimpleNamespace(context_length=1_048_576),
            client=SimpleNamespace(base_url="acp://antigravity"),
        )
        assert _budget_for_agent(agent).turn_budget == ANTIGRAVITY_TURN_BUDGET_CHARS
