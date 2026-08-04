from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from agent.antigravity_session import (
    INLINE_PROMPT_LIMIT_BYTES,
    AntigravityConversation,
    _truncate_utf8,
)
from agent.copilot_acp_client import CopilotACPClient
from agent.model_metadata import get_model_context_length


def _messages(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


def test_second_turn_uses_server_conversation_and_only_new_input(monkeypatch):
    conversation = AntigravityConversation()
    calls: list[dict] = []

    def fake_execute(prompt_text, **kwargs):
        calls.append({"prompt": prompt_text, **kwargs})
        return "answer", "", f"conversation-{len(calls)}"

    monkeypatch.setattr(conversation, "_execute", fake_execute)
    first = _messages(("system", "rules"), ("user", "first question"))
    conversation.run("FULL FIRST PROMPT WITH TOOLS", messages=first, model="gemini")

    second = first + _messages(
        ("assistant", "answer"),
        ("user", "second question"),
    )
    conversation.run("FULL SECOND PROMPT", messages=second, model="gemini")

    assert calls[0]["conversation_id"] is None
    assert calls[0]["prompt"] == "FULL FIRST PROMPT WITH TOOLS"
    assert calls[1]["conversation_id"] == "conversation-1"
    assert "second question" in calls[1]["prompt"]
    assert "first question" not in calls[1]["prompt"]
    assert "answer" not in calls[1]["prompt"]


def test_prefix_change_starts_fresh_after_reset_or_compression(monkeypatch):
    conversation = AntigravityConversation()
    calls: list[str | None] = []

    def fake_execute(prompt_text, **kwargs):
        calls.append(kwargs["conversation_id"])
        return "answer", "", "cid"

    monkeypatch.setattr(conversation, "_execute", fake_execute)
    conversation.run(
        "first full",
        messages=_messages(("system", "rules"), ("user", "first")),
        model="gemini",
    )
    conversation.run(
        "compressed full",
        messages=_messages(("system", "compressed summary"), ("user", "latest")),
        model="gemini",
    )
    assert calls == [None, None]


def test_conversation_state_is_instance_local(monkeypatch):
    left = AntigravityConversation()
    right = AntigravityConversation()
    left_calls: list[str | None] = []
    right_calls: list[str | None] = []

    monkeypatch.setattr(
        left,
        "_execute",
        lambda prompt_text, **kwargs: (
            left_calls.append(kwargs["conversation_id"]) or "left",
            "",
            "left-cid",
        ),
    )
    monkeypatch.setattr(
        right,
        "_execute",
        lambda prompt_text, **kwargs: (
            right_calls.append(kwargs["conversation_id"]) or "right",
            "",
            "right-cid",
        ),
    )
    identical = _messages(("system", "same"), ("user", "same"))
    left.run("same", messages=identical, model="gemini")
    right.run("same", messages=identical, model="gemini")
    assert left_calls == [None]
    assert right_calls == [None]


def test_utf8_prompt_bound_preserves_head_and_tail():
    text = "HEAD:" + ("🙂" * 100_000) + ":TAIL"
    bounded = _truncate_utf8(text)
    assert len(bounded.encode("utf-8")) <= INLINE_PROMPT_LIMIT_BYTES
    assert bounded.startswith("HEAD:")
    assert bounded.endswith(":TAIL")
    assert "middle context omitted" in bounded


def test_client_routes_only_antigravity_marker_to_session_mode(monkeypatch):
    messages = _messages(("user", "hello"))

    antigravity = CopilotACPClient(base_url="acp://antigravity")
    agy_run = Mock(return_value=("AGY_OK", ""))
    monkeypatch.setattr(antigravity._antigravity_conversation, "run", agy_run)
    monkeypatch.setattr(
        antigravity,
        "_run_prompt",
        Mock(side_effect=AssertionError("generic ACP must not run")),
    )
    result = antigravity.chat.completions.create(
        model="gemini-3.6-flash-high",
        messages=messages,
    )
    assert result.choices[0].message.content == "AGY_OK"
    assert agy_run.call_args.kwargs["messages"] == messages

    copilot = CopilotACPClient(base_url="acp://copilot")
    generic_run = Mock(return_value=("COPILOT_OK", ""))
    monkeypatch.setattr(copilot, "_run_prompt", generic_run)
    monkeypatch.setattr(
        copilot._antigravity_conversation,
        "run",
        Mock(side_effect=AssertionError("AGY must not run")),
    )
    result = copilot.chat.completions.create(model="copilot-acp", messages=messages)
    assert result.choices[0].message.content == "COPILOT_OK"
    generic_run.assert_called_once()


def test_gemini_spawn_context_window_is_one_mib_tokens():
    assert get_model_context_length(
        "gemini-3.6-flash-high",
        base_url="acp://antigravity",
        provider="copilot-acp",
    ) == 1_048_576
