"""Focused contracts for the distinct Prime ACP transport backend."""

from __future__ import annotations

import io
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

import hermes_cli.model_switch as model_switch

from agent.copilot_acp_client import (
    CopilotACPClient,
    PrimeACPConversation,
    _PrimeACPState,
    is_acp_stdio_runtime,
)
from agent.auxiliary_client import resolve_provider_client
from agent.portal_tags import reset_conversation_context, set_conversation_context
from agent.secret_scope import (
    UnscopedSecretError,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_cli.auth import (
    AuthError,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
)
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


PRIME_KWARGS = {
    "api_key": "prime-acp",
    "base_url": "acp://prime",
    "command": "/opt/prime/bin/prime",
    "args": ["agent", "acp"],
    "acp_cwd": "/tmp",
}


def _prime_client() -> CopilotACPClient:
    return CopilotACPClient(**PRIME_KWARGS)


def test_prime_marker_is_stdio_and_requires_explicit_runtime(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/wrong/copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")

    assert is_acp_stdio_runtime(provider="custom", base_url="acp://prime")
    assert not is_acp_stdio_runtime(provider="custom", base_url="acp://prime-evil")
    with pytest.raises(ValueError, match="Prime ACP requires an explicit command and args"):
        CopilotACPClient(base_url="acp://prime")
    with pytest.raises(ValueError, match="Prime ACP requires an explicit command and args"):
        CopilotACPClient(base_url="acp://prime", command="prime", args=[])

    client = _prime_client()
    assert client._acp_command == "/opt/prime/bin/prime"
    assert client._acp_args == ["agent", "acp"]


def test_prime_runtime_resolution_never_inherits_copilot_defaults(monkeypatch):
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://prime")
    monkeypatch.delenv("HERMES_COPILOT_ACP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_COPILOT_ACP_ARGS", raising=False)
    monkeypatch.setenv("COPILOT_CLI_PATH", "/wrong/copilot")

    status = get_external_process_provider_status("copilot-acp")
    assert status["configured"] is False
    assert status["command"] == ""
    assert status["args"] == []
    with pytest.raises(AuthError) as exc:
        resolve_external_process_provider_credentials("copilot-acp")
    assert exc.value.code == "missing_prime_acp_runtime"


def test_prime_runtime_resolution_uses_profile_scope_and_fails_closed(
    monkeypatch, tmp_path
):
    command = tmp_path / "prime-agent"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://cursor")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/wrong/process-command")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")

    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            resolve_external_process_provider_credentials("copilot-acp")

        token = set_secret_scope(
            {
                "COPILOT_ACP_BASE_URL": "acp://prime",
                "HERMES_COPILOT_ACP_COMMAND": str(command),
                "HERMES_COPILOT_ACP_ARGS": (
                    "--mode acp --provider zai --model glm-5.2 --thinking xhigh"
                ),
            }
        )
        try:
            creds = resolve_external_process_provider_credentials(
                "copilot-acp",
                target_model="prime:anthropic:claude-sonnet-5:high",
            )
            status = get_external_process_provider_status("copilot-acp")
        finally:
            reset_secret_scope(token)
    finally:
        set_multiplex_active(False)

    assert creds["base_url"] == "acp://prime"
    assert creds["command"] == str(command)
    assert creds["args"] == [
        "--mode", "acp", "--provider", "anthropic",
        "--model", "claude-sonnet-5", "--thinking", "high",
    ]
    assert status["configured"] is True
    assert status["command"] == str(command)


def test_prime_model_switch_is_labeled_and_validated_by_runtime(monkeypatch):
    selector = "prime:anthropic:claude-sonnet-5:high"
    command = "/opt/prime/bin/prime"
    args = [
        "--mode", "acp", "--provider", "anthropic",
        "--model", "claude-sonnet-5", "--thinking", "high",
    ]
    monkeypatch.setitem(
        model_switch.DIRECT_ALIASES,
        "prime-sonnet5-high",
        model_switch.DirectAlias(
            model=selector,
            provider="copilot-acp",
            base_url="acp://prime",
            reasoning_effort="high",
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "copilot-acp",
            "api_key": "prime-acp",
            "base_url": "acp://prime",
            "api_mode": "chat_completions",
            "command": command,
            "args": args,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_a, **_k: pytest.fail(
            "Prime selectors must not use the GitHub Copilot model catalog"
        ),
    )
    monkeypatch.setattr(
        model_switch, "get_model_capabilities", lambda *_a, **_k: None
    )
    monkeypatch.setattr(model_switch, "get_model_info", lambda *_a, **_k: None)
    monkeypatch.setattr(
        model_switch, "_check_hermes_model_warning", lambda *_a, **_k: None
    )

    result = model_switch.switch_model(
        raw_input="prime-sonnet5-high",
        current_provider="copilot-acp",
        current_model="prime:zai:glm-5.2:xhigh",
        current_base_url="acp://prime",
        current_api_key="prime-acp",
    )

    assert result.success is True
    assert result.new_model == selector
    assert result.provider_label == "Prime Agent"
    assert result.warning_message == ""
    assert result.command == command
    assert result.args == args


def test_prime_model_switch_fails_closed_when_runtime_resolution_fails(monkeypatch):
    selector = "prime:anthropic:claude-sonnet-5:high"
    monkeypatch.setitem(
        model_switch.DIRECT_ALIASES,
        "prime-sonnet5-high",
        model_switch.DirectAlias(
            model=selector,
            provider="copilot-acp",
            base_url="acp://prime",
            reasoning_effort="high",
        ),
    )

    def _fail_runtime(**_kwargs):
        raise ValueError("Prime ACP requires an explicit command and args")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _fail_runtime
    )

    result = model_switch.switch_model(
        raw_input="prime-sonnet5-high",
        current_provider="copilot-acp",
        current_model="prime:zai:glm-5.2:xhigh",
        current_base_url="acp://prime",
        current_api_key="prime-acp",
    )

    assert result.success is False
    assert result.target_provider == "copilot-acp"
    assert "Prime ACP requires an explicit command and args" in result.error_message


def test_prime_runtime_resolution_requires_and_returns_exact_argv(monkeypatch, tmp_path):
    command = tmp_path / "prime-agent"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://prime")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(command))
    monkeypatch.setenv(
        "HERMES_COPILOT_ACP_ARGS",
        "--mode acp --provider zai --model glm-5.2 --thinking xhigh",
    )

    creds = resolve_external_process_provider_credentials("copilot-acp")
    assert creds == {
        "provider": "copilot-acp",
        "api_key": "prime-acp",
        "base_url": "acp://prime",
        "command": str(command),
        "args": [
            "--mode", "acp", "--provider", "zai", "--model", "glm-5.2",
            "--thinking", "xhigh",
        ],
        "source": "prime-process",
    }


@pytest.mark.parametrize(
    ("target_model", "provider", "model", "thinking"),
    [
        ("prime:anthropic:claude-fable-5:high", "anthropic", "claude-fable-5", "high"),
        ("prime:anthropic:claude-opus-5:max", "anthropic", "claude-opus-5", "max"),
        ("prime:anthropic:claude-sonnet-5:high", "anthropic", "claude-sonnet-5", "high"),
    ],
)
def test_prime_direct_alias_selects_inference_without_replacing_prime_runtime(
    monkeypatch, tmp_path, target_model, provider, model, thinking
):
    command = tmp_path / "prime-agent"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://prime")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(command))
    monkeypatch.setenv(
        "HERMES_COPILOT_ACP_ARGS",
        "--mode acp --provider zai --model glm-5.2 --thinking xhigh",
    )

    creds = resolve_external_process_provider_credentials(
        "copilot-acp", target_model=target_model
    )

    assert creds["provider"] == "copilot-acp"
    assert creds["base_url"] == "acp://prime"
    assert creds["source"] == "prime-process"
    assert creds["command"] == str(command)
    assert creds["args"] == [
        "--mode", "acp", "--provider", provider, "--model", model,
        "--thinking", thinking,
    ]

    runtime = resolve_runtime_provider(
        requested="copilot-acp",
        target_model=target_model,
    )
    assert runtime["provider"] == "copilot-acp"
    assert runtime["base_url"] == "acp://prime"
    assert runtime["command"] == str(command)
    assert runtime["args"] == creds["args"]


def test_prime_auxiliary_client_forwards_selected_runtime_model(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

    def _resolve(provider, **kwargs):
        captured["provider"] = provider
        captured["target_model"] = kwargs.get("target_model")
        return {
            "provider": "copilot-acp",
            "api_key": "prime-acp",
            "base_url": "acp://prime",
            "command": "/opt/prime/bin/prime-agent",
            "args": [
                "--mode", "acp", "--provider", "anthropic",
                "--model", "claude-opus-5", "--thinking", "max",
            ],
            "source": "prime-process",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        _resolve,
    )
    monkeypatch.setattr("agent.copilot_acp_client.CopilotACPClient", _FakeClient)

    selected = "prime:anthropic:claude-opus-5:max"
    client, model = resolve_provider_client("copilot-acp", selected)

    assert isinstance(client, _FakeClient)
    assert model == selected
    assert captured["provider"] == "copilot-acp"
    assert captured["target_model"] == selected
    assert captured["client_kwargs"] == {
        "api_key": "prime-acp",
        "base_url": "acp://prime",
        "command": "/opt/prime/bin/prime-agent",
        "args": [
            "--mode", "acp", "--provider", "anthropic",
            "--model", "claude-opus-5", "--thinking", "max",
        ],
    }


@pytest.mark.parametrize(
    "target_model",
    [
        "prime:anthropic:claude-opus-5:ultracode",
        "prime:anthropic:claude-opus-5",
        "prime:anthropic:../escape:max",
    ],
)
def test_prime_direct_alias_rejects_invalid_runtime_selector(
    monkeypatch, tmp_path, target_model
):
    command = tmp_path / "prime-agent"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://prime")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(command))
    monkeypatch.setenv(
        "HERMES_COPILOT_ACP_ARGS",
        "--mode acp --provider zai --model glm-5.2 --thinking xhigh",
    )

    with pytest.raises(AuthError) as exc:
        resolve_external_process_provider_credentials(
            "copilot-acp", target_model=target_model
        )
    assert exc.value.code == "invalid_prime_runtime_model"


@pytest.mark.parametrize(
    "runtime_args",
    [
        "--mode acp --provider zai --model glm-5.2",
        "--mode acp --provider zai --provider anthropic --model glm-5.2 --thinking xhigh",
        "--mode acp --mode acp --provider zai --model glm-5.2 --thinking xhigh",
    ],
)
def test_prime_direct_alias_rejects_missing_or_duplicate_runtime_flags(
    monkeypatch, tmp_path, runtime_args
):
    command = tmp_path / "prime-agent"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp://prime")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(command))
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", runtime_args)

    with pytest.raises(AuthError) as exc:
        resolve_external_process_provider_credentials(
            "copilot-acp",
            target_model="prime:anthropic:claude-opus-5:max",
        )
    assert exc.value.code == "invalid_prime_acp_runtime"


def test_prime_prompt_boundary_and_final_text_is_opaque():
    client = _prime_client()
    client._owns_prime_conversation = False
    run = Mock(
        return_value=(
            'Prime final <tool_call>{"id":"x","type":"function",'
            '"function":{"name":"terminal","arguments":"{}"}}</tool_call>',
            "private reasoning",
        )
    )
    client._prime_conversation.run = run
    token = set_conversation_context("discord:dm:123")
    try:
        result = client.chat.completions.create(
            model="ignored-model",
            messages=[
                {"role": "system", "content": "SECRET HERMES SYSTEM"},
                {"role": "user", "content": "old request"},
                {"role": "assistant", "content": "old answer"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "newest visible request"},
                        {"type": "image_url", "image_url": {"url": "ignored"}},
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "terminal", "description": "SECRET TOOL"},
                }
            ],
            tool_choice="auto",
        )
    finally:
        reset_conversation_context(token)

    assert run.call_args.args == ("newest visible request",)
    assert run.call_args.kwargs["state_key"] == "discord:dm:123"
    message = result.choices[0].message
    assert message.content.startswith("Prime final <tool_call>")
    assert message.tool_calls == []
    assert result.choices[0].finish_reason == "stop"


class _LiveProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = []
        self.stderr = []
        self.returncode = None

    def poll(self):
        return self.returncode


class _GracefulProcess(_LiveProcess):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def wait(self, timeout=None):
        self.events.append(f"wait:{timeout}")
        assert self.stdin.closed, "Prime must receive stdin EOF before wait"
        self.returncode = 0
        return 0

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")


def _wire_fake_acp(conversation: PrimeACPConversation, client: CopilotACPClient):
    processes: list[_LiveProcess] = []
    calls: list[tuple[str, dict]] = []
    next_session = 0

    def spawn():
        process = _LiveProcess()
        processes.append(process)
        return process

    def request(state, method, params, **kwargs):
        nonlocal next_session
        calls.append((method, params))
        if method == "initialize":
            return {"agentCapabilities": {"loadSession": True}}
        if method == "session/new":
            next_session += 1
            return {"sessionId": f"prime-session-{next_session}"}
        if method == "session/prompt":
            kwargs["text_parts"].append(f"reply:{params['prompt'][0]['text']}")
            return {"stopReason": "end_turn"}
        if method == "session/load":
            return {"sessionId": params["sessionId"]}
        raise AssertionError(method)

    client._spawn_acp_process = spawn
    conversation._request = request
    return processes, calls


def test_sequential_prompts_reuse_one_live_process_and_session():
    client = _prime_client()
    conversation = PrimeACPConversation()
    processes, calls = _wire_fake_acp(conversation, client)

    first = conversation.run("set x", state_key="discord:thread:7", timeout_seconds=1, client=client)
    second = conversation.run("read x", state_key="discord:thread:7", timeout_seconds=1, client=client)

    assert first[0] == "reply:set x"
    assert second[0] == "reply:read x"
    assert len(processes) == 1
    assert [method for method, _ in calls].count("session/new") == 1
    prompts = [params for method, params in calls if method == "session/prompt"]
    assert [item["sessionId"] for item in prompts] == [
        "prime-session-1",
        "prime-session-1",
    ]


def test_conversation_keys_get_isolated_processes_and_sessions():
    client = _prime_client()
    conversation = PrimeACPConversation()
    processes, calls = _wire_fake_acp(conversation, client)

    conversation.run("DM", state_key="discord:dm:1", timeout_seconds=1, client=client)
    conversation.run("thread", state_key="discord:thread:99", timeout_seconds=1, client=client)
    conversation.run("DM again", state_key="discord:dm:1", timeout_seconds=1, client=client)

    assert len(processes) == 2
    prompts = [params for method, params in calls if method == "session/prompt"]
    assert [item["sessionId"] for item in prompts] == [
        "prime-session-1",
        "prime-session-2",
        "prime-session-1",
    ]


def test_graceful_close_sends_eof_before_bounded_wait():
    process = _GracefulProcess()
    state = _PrimeACPState(key="discord:dm:close", process=process, session_id="sid")

    PrimeACPConversation._close_process(state)

    assert state.process is None
    assert process.stdin.closed
    assert process.events == ["wait:2.0"]


def test_recreation_without_exact_load_support_fails_instead_of_new_session():
    client = _prime_client()
    conversation = PrimeACPConversation()
    state = _PrimeACPState(key="discord:dm:resume", session_id="saved-exact-id")
    process = _GracefulProcess()
    client._spawn_acp_process = Mock(return_value=process)
    conversation._request = Mock(return_value={"agentCapabilities": {}})

    with pytest.raises(RuntimeError, match="does not advertise session/load"):
        conversation._start_state(state, timeout_seconds=1, client=client)

    assert state.session_id == "saved-exact-id"
    assert state.process is None
    assert process.stdin.closed


def test_request_local_client_shares_primary_prime_conversation():
    primary = _prime_client()
    request = _prime_client()
    agent = object.__new__(AIAgent)
    agent.provider = "custom"
    agent._client_kwargs = dict(PRIME_KWARGS)
    agent._ensure_primary_openai_client = Mock(return_value=primary)
    agent._openai_client_lock = Mock(return_value=nullcontext())
    agent._create_openai_client = Mock(return_value=request)

    actual = agent._create_request_openai_client(reason="test", api_kwargs={})

    assert actual is request
    assert request._prime_conversation is primary._prime_conversation
    assert request._owns_prime_conversation is False
