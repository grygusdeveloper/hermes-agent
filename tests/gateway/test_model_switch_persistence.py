"""Tests that gateway /model switch persists across messages.

The gateway /model command stores session overrides in
``_session_model_overrides``.  These must:

1. Be applied in ``run_sync()`` so the next agent uses the switched model.
2. Not be mistaken for fallback activation (which evicts the cached agent).
3. Survive across multiple messages until /reset clears them.

Tests exercise the real ``_apply_session_model_override()`` and
``_is_intentional_model_switch()`` methods on ``GatewayRunner``.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    """Create a minimal GatewayRunner with stubbed internals."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._effective_model = None
    runner._effective_provider = None
    runner.session_store = MagicMock()
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    return runner


# ---------------------------------------------------------------------------
# Tests: _apply_session_model_override
# ---------------------------------------------------------------------------


class TestApplySessionModelOverride:
    """Verify _apply_session_model_override replaces config defaults."""

    def test_override_replaces_all_fields(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4-turbo",
            "provider": "openrouter",
            "api_key": "or-key-123",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4-turbo"
        assert rt["provider"] == "openrouter"
        assert rt["api_key"] == "or-key-123"
        assert rt["base_url"] == "https://openrouter.ai/api/v1"
        assert rt["api_mode"] == "chat_completions"

    def test_no_override_returns_originals(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        orig_model = "anthropic/claude-sonnet-4"
        orig_rt = {"provider": "anthropic", "api_key": "key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"}

        model, rt = runner._apply_session_model_override(sk, orig_model, dict(orig_rt))

        assert model == orig_model
        assert rt == orig_rt

    def test_credentialless_override_clears_foreign_api_mode(self):
        """A credential-less claude-code override on an openai-codex config runtime must not
        inherit ``codex_responses`` (ClaudeCodeClient has no ``responses`` API)."""
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._session_model_overrides[sk] = {
            "model": "claude-sonnet-5",
            "provider": "claude-code",
            "base_url": "acp://claude-code",
            "api_key": None,
            "api_mode": None,
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "gpt-5.6-sol",
            {
                "provider": "openai-codex",
                "api_key": "codex-token",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
            },
        )

        assert model == "claude-sonnet-5"
        assert rt["provider"] == "claude-code"
        assert rt["base_url"] == "acp://claude-code"
        assert rt["api_mode"] is None

    def test_same_provider_override_keeps_runtime_api_mode(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._session_model_overrides[sk] = {
            "model": "gpt-5.6-terra",
            "provider": "openai-codex",
            "api_mode": None,
        }

        _model, rt = runner._apply_session_model_override(
            sk, "gpt-5.6-sol", {"provider": "openai-codex", "api_mode": "codex_responses"},
        )

        assert rt["api_mode"] == "codex_responses"


def test_agent_init_forces_chat_completions_for_claude_code():
    from run_agent import AIAgent

    agent = AIAgent(
        model="claude-sonnet-5",
        provider="claude-code",
        base_url="acp://claude-code",
        api_key="claude-code",
        api_mode="codex_responses",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        assert agent.api_mode == "chat_completions"
    finally:
        try:
            agent.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests: _is_intentional_model_switch
# ---------------------------------------------------------------------------


class TestIsIntentionalModelSwitch:
    """The fallback-eviction check must not evict a session whose model differs from the config
    default for a reason the system produced: a /model override, or the Nous gateway moving the
    session off the ``nous/welcome`` alias the config still carries."""

    def test_matches_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        agent = SimpleNamespace(model="gpt-5.4")
        assert runner._is_intentional_model_switch(sk, agent, "openai/gpt-5") is True

    def test_server_model_switch_off_the_welcome_alias_is_intentional(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        # apply_model_switch moved the session and recorded the move (alias -> backing).
        agent = SimpleNamespace(model="z-ai/glm-5.3-flash", _nous_model_switch=("nous/welcome", "z-ai/glm-5.3-flash"))
        assert runner._is_intentional_model_switch(sk, agent, "nous/welcome") is True
        # A config that names something else is real drift, not the server's move.
        assert runner._is_intentional_model_switch(sk, agent, "openai/gpt-5") is False
        # A later fallback onto a third model is drift too, even with the config still on the alias.
        agent.model = "fallback/model"
        assert runner._is_intentional_model_switch(sk, agent, "nous/welcome") is False

    def test_plain_drift_is_not_intentional(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        agent = SimpleNamespace(model="fallback/model")
        assert runner._is_intentional_model_switch(sk, agent, "primary/model") is False


class TestOneTurnModelOverrideRestore:
    """Verify gateway one-turn overrides restore previous session state."""

    def test_restores_previous_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        previous = {
            "model": "old/model",
            "provider": "openrouter",
            "api_key": "old-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }
        runner._session_model_overrides[sk] = previous

        snapshot = runner._snapshot_session_model_override(sk)
        runner._session_model_overrides[sk] = {
            "model": "temp/model",
            "provider": "anthropic",
        }

        runner._restore_session_model_override(sk, snapshot)

        assert runner._session_model_overrides[sk] == previous

    def test_restores_reasoning_state_owned_before_one_turn_alias(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._set_session_reasoning_override(sk, {"enabled": True, "effort": "medium"})

        snapshot = runner._snapshot_session_model_override(sk)
        runner._apply_model_alias_reasoning_override(sk, "xhigh")
        assert runner._session_state(sk).conversation.reasoning_override_owner == "model_alias"

        runner._restore_session_model_override(sk, snapshot)

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "medium"}
        assert conversation.reasoning_override_owner is None

    def test_restore_brings_back_the_alias_preset_a_one_turn_model_retired(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._apply_model_alias_reasoning_override(sk, "xhigh")

        snapshot = runner._snapshot_session_model_override(sk)
        runner._apply_model_alias_reasoning_override(sk, "")  # plain --once model
        assert runner._session_state(sk).conversation.reasoning_override is None

        runner._restore_session_model_override(sk, snapshot)

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "xhigh"}
        assert conversation.reasoning_override_owner == "model_alias"

    def test_explicit_reasoning_set_during_a_one_shot_window_survives_restore(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._evict_cached_agent = lambda _sk: None

        snapshot = runner._snapshot_session_model_override(sk)
        runner._set_reasoning_override(sk, {"enabled": True, "effort": "high"})  # /reasoning high

        runner._restore_session_model_override(sk, snapshot)

        assert runner._session_state(sk).conversation.reasoning_override == {"enabled": True, "effort": "high"}


class TestModelAliasReasoningLifecycle:
    def test_alias_transition_then_ordinary_model_clears_only_alias_owned_state(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._apply_model_alias_reasoning_override(sk, "high")
        runner._apply_model_alias_reasoning_override(sk, "xhigh")
        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "xhigh"}
        assert conversation.reasoning_override_owner == "model_alias"

        runner._apply_model_alias_reasoning_override(sk, None)
        assert conversation.reasoning_override is None
        assert conversation.reasoning_override_owner is None

    def test_ordinary_model_switch_preserves_explicit_reasoning_choice(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._set_session_reasoning_override(sk, {"enabled": True, "effort": "high"})

        runner._apply_model_alias_reasoning_override(sk, None)

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "high"}
        assert conversation.reasoning_override_owner is None

    def test_explicit_reasoning_takes_ownership_from_an_alias_preset(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._evict_cached_agent = lambda _sk: None
        runner._apply_model_alias_reasoning_override(sk, "xhigh")

        runner._set_reasoning_override(sk, {"enabled": True, "effort": "low"})  # /reasoning low
        runner._apply_model_alias_reasoning_override(sk, None)  # then a plain /model pick

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "low"}
        assert conversation.reasoning_override_owner is None


class TestOneTurnNeverPersisted:
    """/model --once must never write through to the session store.

    Regression guard for the #29923 review defect: the original
    implementation wrote the once-override through set_model_override, so a
    gateway restart before the finally-restore rehydrated a supposedly
    one-turn model permanently. Drives the real _handle_model_command with
    a mocked switch pipeline and asserts on the store boundary.
    """

    @staticmethod
    def _runner_with_store(tmp_path, monkeypatch):
        import hermes_yaml as _yaml

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner
        from hermes_cli.model_switch import ModelSwitchResult

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            _yaml.safe_dump(
                {"model": {"default": "old-model", "provider": "openrouter"}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.5",
                target_provider="openrouter",
                provider_changed=False,
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
                runtime_capabilities={"openai_native_compaction": True},
                provider_label="OpenRouter",
            ),
        )
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._pending_one_turn_model_restores = {}
        runner._running_agents = {}
        # async_session_store is a property over session_store; install the
        # mock behind the private cache attribute it reads.
        _store = MagicMock()
        _store.set_model_override = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _event(text):
        from gateway.platforms.event import MessageEvent, MessageType

        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=_make_source(),
        )

    @pytest.mark.asyncio
    async def test_once_skips_session_store_write_through(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())

        result = await runner._handle_model_command(
            self._event("/model gpt-5.5 --once")
        )

        assert result is not None and "gpt-5.5" in result
        # In-memory override installed for the next turn + restore queued...
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        assert runner._session_model_overrides[sk]["capabilities"] == {
            "openai_native_compaction": True
        }
        assert sk in runner._pending_one_turn_model_restores
        # ...but NEVER written through to the persistent session store.
        runner.async_session_store.set_model_override.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repeated_once_keeps_the_earliest_restore_target(self, tmp_path, monkeypatch):
        """`/model X --once` then `/model Y --once` before any turn: the pending snapshot must still
        be the user's standing override (none here), not X — otherwise slot cleanup would make the
        first temporary model permanent."""
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())

        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))

        # The second producer call snapshotted the live gpt-5.5 override; the pending restore
        # must still be the ORIGINAL "no override" state.
        assert runner._pending_one_turn_model_restores[sk]["had_override"] is False

    @staticmethod
    def _switch_to_reasoning_alias(monkeypatch):
        from hermes_cli.model_switch import ModelSwitchResult

        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.6-sol",
                target_provider="openai-codex",
                reasoning_effort="xhigh",
                provider_label="OpenAI Codex",
            ),
        )

    @pytest.mark.asyncio
    async def test_reasoning_alias_global_is_rejected_before_state_change(self, tmp_path, monkeypatch):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        self._switch_to_reasoning_alias(monkeypatch)
        config_path = tmp_path / ".hermes" / "config.yaml"
        before = config_path.read_text()

        result = await runner._handle_model_command(self._event("/model sol-xhigh --global"))

        assert result is not None and "--global" in result
        assert not runner._session_model_overrides
        assert config_path.read_text() == before
        runner.async_session_store.set_model_override.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reasoning_alias_session_path_applies_and_persists_effort(self, tmp_path, monkeypatch):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        self._switch_to_reasoning_alias(monkeypatch)
        sk = build_session_key(_make_source())

        result = await runner._handle_model_command(self._event("/model sol-xhigh --session"))

        assert result is not None and "gpt-5.6-sol" in result and "xhigh" in result
        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "xhigh"}
        assert conversation.reasoning_override_owner == "model_alias"
        persisted = runner.async_session_store.set_model_override.await_args.args[1]
        assert persisted["reasoning_effort"] == "xhigh"

    @pytest.mark.asyncio
    async def test_plain_reasoning_alias_forces_session_scope_when_default_persists(self, tmp_path, monkeypatch):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        self._switch_to_reasoning_alias(monkeypatch)
        monkeypatch.setattr("hermes_cli.model_switch.resolve_persist_behavior", lambda *a, **k: True)
        config_path = tmp_path / ".hermes" / "config.yaml"
        before = config_path.read_text()
        sk = build_session_key(_make_source())

        result = await runner._handle_model_command(self._event("/model sol-xhigh"))

        assert result is not None and "session" in result.lower()
        assert config_path.read_text() == before
        assert runner._session_state(sk).conversation.reasoning_override == {"enabled": True, "effort": "xhigh"}
        persisted = runner.async_session_store.set_model_override.await_args.args[1]
        assert persisted["reasoning_effort"] == "xhigh"

    @pytest.mark.asyncio
    async def test_typed_reasoning_flag_wins_over_the_alias_preset(self, tmp_path, monkeypatch):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        self._switch_to_reasoning_alias(monkeypatch)
        sk = build_session_key(_make_source())

        await runner._handle_model_command(self._event("/model sol-xhigh --session --reasoning low"))

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override == {"enabled": True, "effort": "low"}
        assert conversation.reasoning_override_owner is None
        persisted = runner.async_session_store.set_model_override.await_args.args[1]
        assert "reasoning_effort" not in persisted

    @pytest.mark.asyncio
    async def test_plain_model_after_reasoning_alias_retires_the_preset(self, tmp_path, monkeypatch):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        self._switch_to_reasoning_alias(monkeypatch)
        sk = build_session_key(_make_source())
        await runner._handle_model_command(self._event("/model sol-xhigh --session"))
        assert runner._session_state(sk).conversation.reasoning_override_owner == "model_alias"
        from hermes_cli.model_switch import ModelSwitchResult
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(success=True, new_model="gpt-5.5", target_provider="openrouter"),
        )

        await runner._handle_model_command(self._event("/model gpt-5.5 --session"))

        conversation = runner._session_state(sk).conversation
        assert conversation.reasoning_override is None
        assert conversation.reasoning_override_owner is None

