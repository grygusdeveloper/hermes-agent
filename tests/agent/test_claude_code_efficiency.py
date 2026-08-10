import json


def test_claude_code_result_usage_is_normalized_for_hermes():
    from agent.claude_code_client import _completion_usage
    from agent.claude_code_session import _parse_stream_json_usage
    from agent.usage_pricing import normalize_usage

    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "12345678-1234-1234-1234-123456789abc",
            "result": "OK",
            "total_cost_usd": 0.125,
            "duration_ms": 2100,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 13,
                "cache_creation_input_tokens": 17,
                "cache_read_input_tokens": 19,
                "service_tier": "standard",
            },
        }
    )
    usage = _parse_stream_json_usage(stdout)
    assert usage["prompt_tokens"] == 47
    assert usage["completion_tokens"] == 13
    assert usage["total_tokens"] == 60
    assert usage["cached_tokens"] == 19
    assert usage["cache_write_tokens"] == 17
    assert usage["total_cost_usd"] == 0.125

    canonical = normalize_usage(_completion_usage(usage), provider="claude-code")
    assert canonical.input_tokens == 11
    assert canonical.output_tokens == 13
    assert canonical.cache_read_tokens == 19
    assert canonical.cache_write_tokens == 17


def test_claude_code_prompt_requires_efficient_tool_only_turns():
    from agent.claude_code_client import _format_messages_as_prompt

    prompt = _format_messages_as_prompt(
        [{"role": "user", "content": "Check it"}],
        model="claude-opus-5",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "probe",
                    "description": "Probe once",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert "FINALITY RULE" in prompt
    assert "Do not add narration before or after tool calls" in prompt
    assert "Independent calls may be emitted together" in prompt
    assert "every tool listed below remains available" in prompt


def test_progress_retry_continues_same_session_instead_of_replaying(monkeypatch):
    from agent.claude_code_session import ClaudeCodeSession

    session = ClaudeCodeSession()
    sid = "12345678-1234-1234-1234-123456789abc"
    calls = []

    def fake_execute(prompt, *, session_id, **_kwargs):
        calls.append((prompt, session_id))
        if len(calls) == 1:
            return "I'll inspect that now.", "", sid
        return "Completed result.", "", sid

    monkeypatch.setattr(session, "_execute", fake_execute)
    response, _, returned_sid = session._execute_with_soft_limit_retry(
        "ORIGINAL FULL PROMPT",
        session_id=None,
        model="claude-opus-5",
        effort="low",
        timeout_seconds=30,
        cwd="/tmp",
        env={},
        max_attempts=3,
        had_tools=True,
    )
    assert response == "Completed result."
    assert returned_sid == sid
    assert calls[0] == ("ORIGINAL FULL PROMPT", None)
    assert calls[1][1] == sid
    assert calls[1][0] != "ORIGINAL FULL PROMPT"
    assert "Continue the previous request now" in calls[1][0]


def test_tool_digest_ignores_registry_order():
    from agent.claude_code_client import _tools_digest

    first = {
        "type": "function",
        "function": {"name": "alpha", "parameters": {"type": "object"}},
    }
    second = {
        "type": "function",
        "function": {"name": "beta", "parameters": {"type": "object"}},
    }
    assert _tools_digest([first, second]) == _tools_digest([second, first])
