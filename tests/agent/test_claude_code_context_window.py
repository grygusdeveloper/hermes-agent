"""Claude Code context window from result modelUsage and compression.provider_threshold_tokens
(ported from WP4 7025953856 / 33d79181f5).

The end-to-end case drives a real AIAgent; only the ``claude`` subprocess is the stream-json fixture
from test_claude_code_provider.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_code_protocol import context_window
from agent.context_compressor import ContextCompressor
from tests.agent.test_claude_code_effort import _actor
from tests.agent.test_claude_code_provider import native, scope  # noqa: F401  (fixture)


# ---------------------------------------------------------------------------
# Context window from result modelUsage
# ---------------------------------------------------------------------------


def test_context_window_comes_from_the_serving_model():
    model_usage = {
        "claude-haiku-4-5-20251001": {"inputTokens": 300, "contextWindow": 200_000},
        "claude-opus-5": {"inputTokens": 12, "cacheReadInputTokens": 150_000, "contextWindow": 1_000_000},
    }
    # Without an assistant model, the entry that read the most input is the conversation's.
    assert context_window(model_usage) == 1_000_000
    assert context_window({}) is None and context_window(None) is None
    assert context_window({"claude-opus-5": {"inputTokens": 5}}) is None


def test_context_window_is_the_serving_models_not_the_largest_reader():
    """modelUsage is cumulative over the CLI session, so another model (a refusal fallback, an
    earlier turn) can hold the most input."""
    model_usage = {
        "claude-sonnet-5": {"inputTokens": 900_000, "contextWindow": 200_000},
        "claude-opus-5": {"inputTokens": 5, "contextWindow": 1_000_000},
    }
    assert context_window(model_usage, "claude-opus-5") == 1_000_000
    assert context_window(model_usage, "claude-unknown") == 200_000


@pytest.mark.platforms("posix")
def test_reported_context_window_resizes_the_running_agent(native, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr("agent.model_metadata.save_context_length", lambda *args, **kw: saved.append(args))
    with scope(tmp_path / "a", "effort-owner"):
        agent = _actor(tmp_path)
        try:
            agent.run_conversation(json.dumps({
                "scenario": "text", "model_usage": {"claude-opus-5": {"inputTokens": 9, "contextWindow": 200_000}},
            }))
            compressor = agent.context_compressor
            assert compressor.context_length == 200_000
            assert compressor.threshold_tokens <= 200_000
        finally:
            agent.close()
    assert saved == [("opus", "acp://claude-code", 200_000)]


def _loop_agent(compressor, **extra):
    fields = dict(model="claude-opus-5", provider="claude-code", base_url="acp://claude-code", api_key="claude-code",
                  api_mode="chat_completions", context_compressor=compressor, _config_context_length=None)
    fields.update(extra)
    return SimpleNamespace(**fields)


def _compressor(context_length=1_000_000, provider="claude-code", **kwargs):
    kwargs.setdefault("provider_threshold_tokens", {"claude-code": 250_000})
    with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
        compressor = ContextCompressor(model="claude-opus-5", provider=provider, base_url="acp://claude-code",
                                       quiet_mode=True, **kwargs)
        compressor.context_length  # resolve now, inside the patch
    return compressor


def test_reported_context_window_updates_the_compressor(monkeypatch):
    from agent.turn_usage import _sync_provider_context_window

    saved = []
    monkeypatch.setattr("agent.model_metadata.save_context_length", lambda *args, **kw: saved.append(args))
    compressor = _compressor()
    assert compressor.threshold_tokens == 250_000
    agent = _loop_agent(compressor)
    _sync_provider_context_window(agent, SimpleNamespace(context_window=200_000))
    assert compressor.context_length == 200_000
    assert compressor.threshold_tokens == 150_000  # 75% floor below 512K; the 250K cap does not bind
    assert saved == [("claude-opus-5", "acp://claude-code", 200_000)]
    # Unchanged, absent or configured windows are left alone.
    _sync_provider_context_window(agent, SimpleNamespace(context_window=200_000))
    _sync_provider_context_window(agent, SimpleNamespace())
    _sync_provider_context_window(agent, MagicMock())
    configured = _loop_agent(_compressor(), _config_context_length=1_000_000)
    _sync_provider_context_window(configured, SimpleNamespace(context_window=200_000))
    assert configured.context_compressor.context_length == 1_000_000
    assert len(saved) == 1


def test_context_window_sync_ignores_other_providers(monkeypatch):
    """An OpenAI-compatible SDK usage object keeps unknown response fields; a ``context_window``
    extra from another provider must not resize Hermes."""
    from agent.turn_usage import _sync_provider_context_window

    monkeypatch.setattr("agent.model_metadata.save_context_length", lambda *args, **kw: None)
    compressor = _compressor(provider="zai")
    agent = _loop_agent(compressor, provider="zai", model="glm-5.3", base_url="https://api.z.ai")
    _sync_provider_context_window(agent, SimpleNamespace(context_window=8_192))
    assert compressor.context_length == 1_000_000


# ---------------------------------------------------------------------------
# compression.provider_threshold_tokens
# ---------------------------------------------------------------------------


def test_provider_cap_applies_only_while_its_provider_is_active():
    compressor = _compressor()
    assert compressor.threshold_tokens == 250_000
    compressor.update_model(model="glm-5.3", context_length=1_000_000, provider="zai", base_url="https://api.z.ai")
    zai_threshold = compressor.threshold_tokens
    assert zai_threshold > 250_000
    compressor.update_model(model="claude-opus-5", context_length=1_000_000, provider="claude-code",
                            base_url="acp://claude-code")
    assert compressor.threshold_tokens == 250_000
    # A lower global cap wins; a non-positive or malformed provider cap disables it.
    assert _compressor(threshold_tokens_cap=100_000).threshold_tokens == 100_000
    assert _compressor(provider_threshold_tokens={"claude-code": 0}).threshold_tokens == zai_threshold
    assert _compressor(provider_threshold_tokens={"claude-code": True}).threshold_tokens == zai_threshold
    assert _compressor(provider_threshold_tokens=["claude-code"]).threshold_tokens == zai_threshold
    # Provider keys match case-insensitively.
    assert _compressor(provider_threshold_tokens={"Claude-Code": 300_000}).threshold_tokens == 300_000


def _init_threshold(compression_cfg):
    from run_agent import AIAgent

    cfg = {"agent": {}, "compression": compression_cfg}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=1_000_000),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        agent = AIAgent(model="claude-opus-5", provider="claude-code", base_url="acp://claude-code",
                        api_key="claude-code", quiet_mode=True, skip_context_files=True, skip_memory=True)
        threshold = agent.context_compressor.threshold_tokens
    agent.close()
    return threshold


def test_claude_code_compacts_at_250k_by_default():
    assert _init_threshold({"threshold": 0.5}) == 250_000
    assert _init_threshold({"threshold": 0.5, "provider_threshold_tokens": {"claude-code": 300_000}}) == 300_000
    assert _init_threshold({"threshold": 0.5, "provider_threshold_tokens": {"claude-code": 0}}) == 500_000
