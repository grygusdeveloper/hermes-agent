"""Quiet work turns: narration before tool calls never becomes a chat message.

On Discord every new message notifies the user. Before this change the
progress sentence Claude wrote ahead of a tool call streamed live, so a turn
with 38 tool steps posted (and pinged) dozens of "Checking the logs..." lines.
Now the bridge keeps that prose off the answer stream and hands it to Hermes
as ``delta.commentary``; Hermes routes it to the interim-message rail (off by
default on Discord) and keeps it in the assistant message for history.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.claude_code_session as ccs
from agent.claude_code_client import ClaudeCodeClient, _build_claude_code_request
from agent.claude_code_session import _HERMES_TOOL_PROTOCOL, _StreamGate

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
CALL = '<tool_call>{"id": "c1", "name": "terminal", "arguments": {"command": "ls"}}</tool_call>'
NARRATED = "Checking the logs for the crash.\n\n" + CALL


def _collect(stream):
    content, commentary, calls, finish = [], [], [], []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content.append(delta.content)
        if getattr(delta, "commentary", None):
            commentary.append(delta.commentary)
        if delta.tool_calls:
            calls.extend(delta.tool_calls)
        if chunk.choices[0].finish_reason:
            finish.append(chunk.choices[0].finish_reason)
    return "".join(content), "".join(commentary), calls, finish


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_holds_narration_and_streams_answers():
    out = []
    gate = _StreamGate(out.append)
    for index in range(0, len(NARRATED), 5):
        gate.feed(NARRATED[index:index + 5])
    assert out == []
    gate.finish(NARRATED)
    assert out == [] and gate.held == "Checking the logs for the crash."

    answer = "The crash comes from the loader; the fix is in place and tests pass. " * 12
    out = []
    gate = _StreamGate(out.append)
    for index in range(0, len(answer), 5):
        gate.feed(answer[index:index + 5])
    assert gate.committed and out
    gate.finish(answer)
    assert "".join(out) == answer.strip() and gate.held == ""


def test_with_tools_prose_commits_only_past_the_narration_bound():
    assert ccs._STREAM_COMMIT_CHARS_WITH_TOOLS == 600
    out = []
    gate = _StreamGate(out.append, had_tools=True)
    gate.feed("Short status line that is under the bound. " * 10)  # 440 chars
    assert out == [] and not gate.committed
    gate.feed("Short status line that is under the bound. " * 5)
    assert gate.committed and out


# ---------------------------------------------------------------------------
# Client stream: commentary channel
# ---------------------------------------------------------------------------


def _stream(client, raw, **kwargs):
    with patch.object(client._claude_session, "run", return_value=(raw, "")):
        return _collect(
            client._create_chat_completion(
                model="opus", messages=[{"role": "user", "content": "check"}], stream=True, **kwargs
            )
        )


def test_narration_before_a_call_is_commentary_not_content():
    client = ClaudeCodeClient(cwd="/tmp")
    content, commentary, calls, finish = _stream(client, NARRATED, tools=TOOLS)
    assert content == ""
    assert commentary == "Checking the logs for the crash."
    assert [tc.function.name for tc in calls] == ["terminal"] and finish == ["tool_calls"]

    # Without a live display the same split applies.
    content, commentary, calls, _finish = _stream(client, NARRATED, tools=TOOLS, live_display=False)
    assert content == "" and commentary == "Checking the logs for the crash."
    assert len(calls) == 1

    # The non-streaming message keeps the prose as content (history).
    with patch.object(client._claude_session, "run", return_value=(NARRATED, "")):
        response = client._create_chat_completion(
            model="opus", messages=[{"role": "user", "content": "check"}], tools=TOOLS
        )
    assert response.choices[0].message.content == "Checking the logs for the crash."


def test_a_reply_without_calls_streams_as_the_answer():
    client = ClaudeCodeClient(cwd="/tmp")
    answer = "Nothing is wrong: the crash was a one-off and the service is healthy."
    content, commentary, calls, finish = _stream(client, answer, tools=TOOLS)
    assert content == answer and commentary == "" and calls == [] and finish == ["stop"]


def test_calls_only_reply_has_no_commentary():
    client = ClaudeCodeClient(cwd="/tmp")
    content, commentary, calls, _finish = _stream(client, CALL, tools=TOOLS)
    assert content == "" and commentary == "" and len(calls) == 1


# ---------------------------------------------------------------------------
# Hermes stream consumer: commentary goes to the interim rail only
# ---------------------------------------------------------------------------


def _chunk(**delta):
    fields = {"content": None, "tool_calls": None, "reasoning_content": None, "reasoning": None}
    fields.update(delta)
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(**fields), finish_reason=None)],
        model="claude-opus-5",
        usage=None,
    )


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_commentary_reaches_the_interim_rail_not_the_stream(_mock_close, mock_create):
    from run_agent import AIAgent

    call = SimpleNamespace(
        index=0, id="c1", type="function",
        function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'),
    )
    finish = _chunk(tool_calls=[call])
    finish.choices[0].finish_reason = "tool_calls"
    chunks = [_chunk(commentary="Checking the logs for the crash."), finish]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    streamed, interim = [], []
    agent.stream_delta_callback = streamed.append
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: interim.append(
        (text, already_streamed)
    )

    response = agent._interruptible_streaming_api_call({})

    assert streamed == []
    assert interim == [("Checking the logs for the crash.", False)]
    message = response.choices[0].message
    assert message.content == "Checking the logs for the crash."
    assert [tc.function.name for tc in message.tool_calls] == ["terminal"]


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_commentary_is_dropped_when_interim_messages_are_off(_mock_close, mock_create):
    from run_agent import AIAgent

    finish = _chunk()
    finish.choices[0].finish_reason = "stop"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(
        [_chunk(commentary="Looking at the disk."), _chunk(content="All good."), finish]
    )
    mock_create.return_value = mock_client
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    streamed = []
    agent.stream_delta_callback = streamed.append
    agent.interim_assistant_callback = None

    response = agent._interruptible_streaming_api_call({})

    assert streamed == ["All good."]
    assert response.choices[0].message.content == "Looking at the disk.All good."


# ---------------------------------------------------------------------------
# Prompt: no narration, chat-sized answers
# ---------------------------------------------------------------------------


def test_tool_protocol_asks_for_calls_only():
    assert "Do not narrate tool use" in _HERMES_TOOL_PROTOCOL
    assert "progress sentence is optional" not in _HERMES_TOOL_PROTOCOL


def _system_prompt(platform_line):
    system, _prompt, _images = _build_claude_code_request(
        [
            {"role": "system", "content": f"You are Hermes.\n{platform_line}\nBe helpful."},
            {"role": "user", "content": "hi"},
        ],
        tools=TOOLS,
    )
    return system


def test_chat_platforms_get_a_one_message_answer_budget():
    discord = _system_prompt("Platform: discord")
    assert "Chat delivery (discord)" in discord
    assert "at most 1,500\ncharacters" in discord
    assert "no tables (they do not render)" in discord
    # The rule follows Hermes's own instructions.
    assert discord.index("Hermes system instructions") < discord.index("Chat delivery (discord)")

    telegram = _system_prompt("Platform: telegram")
    assert "Chat delivery (telegram)" in telegram and "3,000" in telegram

    assert "Chat delivery" not in _system_prompt("Platform: cli")
    assert "Chat delivery" not in _system_prompt("Source: Discord thread 123")


def test_warm_process_outlives_a_slow_tool(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", raising=False)
    assert ccs._keepalive_seconds() == 300.0
