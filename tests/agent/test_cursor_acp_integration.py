"""Cursor ACP uses its own binary, model argv and credential boundary through real discovery."""
import sys

import pytest


def test_cursor_runtime_isolated_from_copilot_and_selects_model(tmp_path, monkeypatch):
    from agent.cursor_acp_client import CursorACPClient
    from hermes_cli.auth import resolve_external_process_provider_credentials
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from providers import get_provider_profile

    # A real stdio JSON-RPC child, exercising initialize -> session/new -> prompt.
    child = tmp_path / "agent"
    child.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
assert sys.argv[1:] == ["--model", "cursor-grok-4.6-high", "acp"], sys.argv[1:]
assert "HERMES_COPILOT_ACP_COMMAND" not in os.environ
assert "OPENAI_API_KEY" not in os.environ
for line in sys.stdin:
    req = json.loads(line)
    method = req["method"]
    if method == "session/prompt":
        print(json.dumps({"jsonrpc":"2.0", "method":"session/update", "params":{"update":{"sessionUpdate":"agent_message_chunk", "content":{"text":os.environ["HERMES_HOME"]}}}}), flush=True)
    print(json.dumps({"jsonrpc":"2.0", "id":req["id"], "result": {"sessionId":"cursor-session"} if method == "session/new" else {}}), flush=True)
''')
    child.chmod(0o700)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "/nonexistent/copilot")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-credential")
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + __import__("os").environ.get("PATH", ""))
    profile = get_provider_profile("cursor")
    assert profile is not None and profile.auth_type == "external_process"
    # Provider credentials are resolved via the real generic runtime path, not a hardcoded slug.
    creds = resolve_external_process_provider_credentials("cursor")
    assert creds["base_url"] == "acp://cursor"
    client = profile.create_client(**creds, acp_cwd=str(tmp_path))
    assert isinstance(client, CursorACPClient)
    for name in ("main", "researcher", "main"):
        home = tmp_path / name
        home.mkdir(exist_ok=True)
        token = set_hermes_home_override(home)
        try:
            result = client.chat.completions.create(model="cursor-grok-4.6-high", messages=[{"role": "user", "content": "hello"}], timeout=5)
            assert result.choices[0].message.content == str(home)
        finally:
            reset_hermes_home_override(token)
    with pytest.raises(ValueError, match="valid model ID"):
        client.chat.completions.create(model="--malicious", messages=[], timeout=5)
    client.close()
