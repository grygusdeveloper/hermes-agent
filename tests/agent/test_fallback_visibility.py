from types import SimpleNamespace

import agent.auxiliary_client as auxiliary_client
import agent.chat_completion_helpers as chat_helpers
from agent.chat_completion_helpers import try_activate_fallback
from agent.error_classifier import FailoverReason


class DummyAgent(SimpleNamespace):
    def _try_activate_fallback(self, reason=None):
        return try_activate_fallback(self, reason)


def _make_agent(*, notify_on_fallback=None):
    emitted: list[str] = []
    buffered: list[str] = []

    agent = DummyAgent(
        provider="zai",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        api_mode="chat_completions",
        api_key="primary-key",
        client=None,
        _client_kwargs={},
        _transport_cache={},
        _fallback_chain=[{"provider": "xai-oauth", "model": "grok-4.3"}],
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"provider": "zai"},
        _rate_limited_until=0,
        _config_context_length=None,
        _credential_pool=None,
        context_compressor=None,
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda _model, provider=None: False,
        _anthropic_prompt_cache_policy=lambda **_kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        _replace_primary_openai_client=lambda reason=None: None,
        _buffer_status=buffered.append,
        _emit_status=emitted.append,
    )
    if notify_on_fallback is not None:
        agent.notify_on_fallback = notify_on_fallback
    return agent, emitted, buffered


def _patch_fallback_resolver(monkeypatch):
    def fake_resolve_provider_client(provider, model="", **_kwargs):
        client = SimpleNamespace(base_url="https://api.x.ai/v1/", api_key="fallback-key")
        return client, model

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", fake_resolve_provider_client)
    monkeypatch.setattr(chat_helpers, "get_provider_request_timeout", lambda provider, model: None)


def test_successful_fallback_stays_buffered_by_default(monkeypatch):
    _patch_fallback_resolver(monkeypatch)
    agent, emitted, buffered = _make_agent(notify_on_fallback=False)

    assert try_activate_fallback(agent, FailoverReason.billing) is True

    assert emitted == []
    assert buffered == ["🔄 Provider fallback: zai/glm-5.2 → xai-oauth/grok-4.3 (billing)"]


def test_notify_on_fallback_emits_immediate_status(monkeypatch):
    _patch_fallback_resolver(monkeypatch)
    agent, emitted, buffered = _make_agent(notify_on_fallback=True)

    assert try_activate_fallback(agent, FailoverReason.billing) is True

    assert buffered == []
    assert emitted == ["🔄 Provider fallback: zai/glm-5.2 → xai-oauth/grok-4.3 (billing)"]


def test_notify_on_fallback_env_override(monkeypatch):
    _patch_fallback_resolver(monkeypatch)
    monkeypatch.setenv("HERMES_NOTIFY_ON_FALLBACK", "true")
    agent, emitted, buffered = _make_agent()

    assert try_activate_fallback(agent, FailoverReason.rate_limit) is True

    assert buffered == []
    assert emitted == ["🔄 Provider fallback: zai/glm-5.2 → xai-oauth/grok-4.3 (rate_limit)"]
