"""A reply in Claude's native tool-use XML never runs, never streams, is repaired.

Observed on 2026-09-19: Opus wrote 116 ``<invoke name=...>`` blocks instead of
``<tool_call>`` blocks, nothing recognised them, the reply ran to the CLI's
64k-token output cap (104 KB) and was streamed into Discord as prose.
"""

from __future__ import annotations

import agent.claude_code_session as ccs
from agent.claude_code_session import (
    _CLI_ENV_DEFAULTS,
    _StreamGate,
    _parse_claude_reply,
    _tool_call_repair_prompt,
)

INVOKE = (
    '\n<invoke name="terminal">\n<parameter name="command">ls -la</parameter>\n</invoke>\n'
    '<invoke name="session_search">\n<parameter name="query">report template</parameter>\n</invoke>\n'
)
CALL = '<tool_call>{"id": "c1", "name": "terminal", "arguments": {"command": "ls"}}</tool_call>'


def test_native_xml_calls_are_a_protocol_failure_not_prose():
    reply = _parse_claude_reply("Looking at the prompts.\n" + INVOKE * 40)
    assert reply.broken and reply.executable_calls == [] and reply.calls == []
    assert reply.cleaned == "Looking at the prompts."
    assert "<invoke" not in reply.cleaned
    failure = reply.failures[0]
    assert failure.name == "terminal" and "<invoke>/<parameter> XML" in failure.error
    prompt = _tool_call_repair_prompt(reply)
    assert "Hermes has no native tools" in prompt
    assert "<invoke>, <function_calls> and <parameter> tags do nothing" in prompt


def test_native_xml_after_real_calls_is_just_a_discarded_tail():
    reply = _parse_claude_reply(CALL + "\n" + INVOKE)
    assert not reply.broken and [c["name"] for c in reply.executable_calls] == ["terminal"]
    assert reply.discarded_tail.strip().startswith("<invoke")


def test_native_xml_inside_code_is_an_example():
    text = "Claude Code renders calls as:\n```xml\n" + INVOKE + "```\nThat is not what Hermes uses."
    reply = _parse_claude_reply(text)
    assert not reply.broken and reply.calls == [] and reply.cleaned == text.strip()


def test_gate_stops_streaming_at_native_xml():
    out = []
    gate = _StreamGate(out.append, commit_chars=20)
    prose = "Here is what I found so far in the prompt layers. " * 3
    text = prose + INVOKE * 30
    for index in range(0, len(text), 11):
        gate.feed(text[index:index + 11])
    joined = "".join(out)
    assert joined == prose.strip()
    assert "<invoke" not in joined
    gate.finish(text)
    assert "".join(out) == prose.strip()


def test_child_process_caps_output_tokens(monkeypatch):
    assert _CLI_ENV_DEFAULTS["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
    env = ccs._build_subprocess_env({"PATH": "/usr/bin", "HOME": "/tmp"})
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "32000")
    env = ccs._build_subprocess_env({"PATH": "/usr/bin", "HOME": "/tmp"})
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
