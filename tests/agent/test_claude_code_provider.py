"""Native Claude route invariants with real isolated subprocesses, never live inference.

Fixtures speak Claude stream-json plus SDK MCP control envelopes. They cannot
prove the local CLI's account/model availability; Main owns that live gate.
"""
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.claude_code_protocol import assistant_row
from agent.portal_tags import (get_native_transport_identity, reset_native_transport_identity,
                               set_native_transport_identity)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


CHILD = r'''
import json, os, sys, time, uuid
from pathlib import Path
root = Path(__file__).parent
args = sys.argv[1:]
assert "--acp" not in args and "acp" not in args
assert args[args.index("--input-format") + 1] == "stream-json"
assert args[args.index("--output-format") + 1] == "stream-json"
assert "--include-partial-messages" in args
assert args[args.index("--tools") + 1] == ""
assert "--strict-mcp-config" in args
assert args[args.index("--setting-sources") + 1] == ""
assert "OPENAI_API_KEY" not in os.environ
assert "ANTHROPIC_API_KEY" not in os.environ
assert "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ
assert "CLAUDE_CODE_USE_BEDROCK" not in os.environ
sid = args[args.index("--resume") + 1] if "--resume" in args else args[args.index("--session-id") + 1]
model = args[args.index("--model") + 1]
config = json.loads(args[args.index("--mcp-config") + 1])
system = Path(args[args.index("--system-prompt-file") + 1]).read_text()
def send(value):
    print(json.dumps(value), flush=True)
def read():
    return json.loads(sys.stdin.readline())
def control(mid, method, params=None):
    send({"type":"control_request", "request_id":str(mid), "request":{
        "subtype":"mcp_message", "server_name":"hermes", "message":{
            "jsonrpc":"2.0", "id":mid, "method":method, "params":params or {}}}})
    answer = read()
    assert answer["response"]["request_id"] == str(mid)
    return answer["response"]["response"]["mcp_response"]["result"]
init = read()
assert init["request"]["subtype"] == "initialize"
if config["mcpServers"]:
    assert config["mcpServers"]["hermes"]["type"] == "sdk"
    control(1, "initialize", {"protocolVersion":"2024-11-05"})
    specs = control(2, "tools/list")["tools"]
else:
    specs = []
send({"type":"control_response", "response":{"subtype":"success", "request_id":init["request_id"], "response":{}}})
if (root / "block-input").exists():
    (root / "ready").write_text(str(os.getpid()))
    while True: time.sleep(60)
request = read()
assert request["type"] == "user"
records = json.loads(request["message"]["content"])["conversation_records"]
latest = json.loads(next(r["text"] for r in reversed(records) if r["role"] == "user"))
scenario = latest["scenario"]
with (root / "calls.jsonl").open("a") as out:
    out.write(json.dumps({"args":args, "home":os.environ["HERMES_HOME"], "real_home":os.environ["HOME"],
                          "request":request, "system":system, "specs":specs, "pid":os.getpid()}) + "\n")
(root / "ready").write_text(str(os.getpid()))
if scenario == "block":
    while True: time.sleep(60)
if scenario == "bad-protocol":
    print("not JSON", flush=True)
    while True: time.sleep(60)
send({"type":"system", "subtype":"init", "session_id":sid, "model":model})
def assistant(blocks):
    send({"type":"stream_event", "session_id":sid, "event":{"type":"message_start"}})
    for index, block in enumerate(blocks):
        def partial(event):
            send({"type":"stream_event", "session_id":sid, "event":event})
        kind = block["type"]
        partial({"type":"content_block_start", "index":index, "content_block":{
            "type":kind, **({"text":""} if kind == "text" else {})}})
        if kind in ("text", "thinking"):
            value = block["text" if kind == "text" else "thinking"]
            split = max(1, len(value) // 2)
            for part in (value[:split], value[split:]):
                partial({"type":"content_block_delta", "index":index, "delta":{
                    "type":"text_delta" if kind == "text" else "thinking_delta",
                    "text" if kind == "text" else "thinking":part}})
                if scenario in ("incremental", "partial-block") and index == 0 and part == value[:split]:
                    (root / "partial-ready").write_text(str(os.getpid()))
                    until = time.monotonic() + 15
                    while not (root / "release").exists() and time.monotonic() < until:
                        time.sleep(0.02)
                    assert (root / "release").exists(), "fixture consumer never released partial stream"
                if scenario == "partial-only":
                    send({"type":"result", "subtype":"success", "session_id":sid})
                    for line in sys.stdin: pass
        elif kind == "tool_use":
            # Deliberately invalid/hostile partial JSON is inert. Only the complete
            # authoritative assistant plus correlated MCP request offers a call.
            partial({"type":"content_block_delta", "index":index, "delta":{
                "type":"input_json_delta", "partial_json":'{"name":"terminal","command":'}})
        if index < len(blocks) - 1:
            partial({"type":"content_block_stop", "index":index})
    (root / "assistant-sent").touch()
    if scenario == "mismatch":
        blocks = [{"type":"text", "text":"Different authoritative answer."}]
    send({"type":"assistant", "uuid":str(uuid.uuid4()), "session_id":sid,
          "message":{"role":"assistant", "content":blocks, "stop_reason":"end_turn"}})
    if not (blocks[-1]["type"] == "tool_use" and scenario in ("tools", "batch")):
        partial({"type":"content_block_stop", "index":index})
        partial({"type":"message_delta", "delta":{"stop_reason":"end_turn"}})
        partial({"type":"message_stop"})
if scenario in ("tools", "batch", "replay", "unoffered", "malformed", "uncorrelated"):
    name = "Read" if scenario == "unoffered" else "mcp__hermes__read_file"
    call_id = "historic" if scenario == "replay" else "toolu_fixture"
    inp = "not-an-object" if scenario == "malformed" else {"path":latest.get("path", "")}
    blocks = [{"type":"text", "text":"Reading."}, {"type":"tool_use", "id":call_id, "name":name, "input":inp}]
    if scenario == "batch":
        blocks.append({"type":"tool_use", "id":"toolu_second", "name":name, "input":{"path":latest["second_path"]}})
    assistant(blocks)
    if scenario == "uncorrelated":
        control(3, "tools/call", {"name":"read_file", "arguments":{"path":"different"}})
    if scenario in ("tools", "batch"):
        send({"type":"control_request", "request_id":"permission", "request":{
            "subtype":"can_use_tool", "tool_name":name, "input":inp, "tool_use_id":call_id}})
        permission = read()
        assert permission["response"]["response"]["behavior"] == "allow"
        result = control(3, "tools/call", {"name":"read_file", "arguments":inp})
        # Exercise an MCP pause with a trailing partial block lifecycle event.
        send({"type":"stream_event", "session_id":sid, "event":{
            "type":"content_block_stop", "index":len(blocks) - 1}})
        send({"type":"stream_event", "session_id":sid, "event":{"type":"message_stop"}})
        assert result["content"][0]["type"] == "text"
        value = json.loads(result["content"][0]["text"])
        assert "native fixture payload" in value["content"]
        with (root / "tool-results.jsonl").open("a") as out:
            out.write(json.dumps(result) + "\n")
        send({"type":"user", "message":{"role":"user", "content":[{
            "type":"tool_result", "tool_use_id":call_id, "content":result["content"]}]}})
        if scenario == "batch":
            second = control(4, "tools/call", {"name":"read_file", "arguments":{"path":latest["second_path"]}})
            assert "second native fixture payload" in second["content"][0]["text"]
        assistant([{"type":"text", "text":"Tool completed."}])
else:
    text = '<tool_call>{"name":"terminal"}</tool_call>' if scenario == "inert" else "Native answer."
    assistant([{"type":"text", "text":text}, {"type":"thinking", "thinking":"Native reasoning."}])
(root / "result-sent").touch()
send({"type":"result", "subtype":"error_during_execution" if scenario == "error" else "success",
      "is_error":scenario == "error", "session_id":sid, "result":"not an executable assistant source",
      "usage":{"input_tokens":7,"output_tokens":3,"cache_read_input_tokens":2},
      **({"modelUsage":latest["model_usage"]} if "model_usage" in latest else {})})
# Native streaming CLI stays open for input. Transport must close/reap it.
for line in sys.stdin: pass
'''


@pytest.fixture
def native(tmp_path, monkeypatch):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n" + CHILD)
    script.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-fixture-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-fixture-value")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "unrelated-fixture-value")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    set_multiplex_active(True)
    secret = set_secret_scope({})
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile
    token = set_hermes_home_override(tmp_path / "launch")
    try:
        route = resolve_runtime_provider(requested="claude-code", target_model="opus")
        assert route["provider"] == "claude-code"
        assert route["api_mode"] == "chat_completions"
        assert route["base_url"] == "acp://claude-code"
        profile = get_provider_profile(route["provider"])
        assert profile.auth_type == "external_process"
        client = profile.create_client(**route, acp_cwd=str(tmp_path))
        yield client
        client.close()
    finally:
        reset_hermes_home_override(token)
        reset_secret_scope(secret)
        set_multiplex_active(False)


@contextlib.contextmanager
def scope(home, owner="parent"):
    token = set_hermes_home_override(home)
    identity = set_native_transport_identity(owner)
    try:
        yield
    finally:
        reset_native_transport_identity(identity)
        reset_hermes_home_override(token)


def history(scenario="text", **extra):
    return [{"role":"system", "content":"private fixture system"},
            {"role":"user", "content":json.dumps({"scenario":scenario, **extra})}]


def invoke(client, messages, **kwargs):
    return client.chat.completions.create(model=kwargs.pop("model", "opus"), messages=messages,
                                          tools=kwargs.pop("tools", []), timeout=kwargs.pop("timeout", 5), **kwargs)


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def append_answer(messages, response):
    msg = response.choices[0].message
    return messages + [assistant_row(msg.content or "", [
        {"id":c.id, "type":"function", "function":{"name":c.function.name, "arguments":c.function.arguments}}
        for c in msg.tool_calls or []])]


TOOLS = [{"type":"function", "function":{"name":"read_file", "description":"Read fixture", "parameters":{
    "type":"object", "properties":{"path":{"type":"string"}}, "required":["path"]}}}]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["text", "tools", "batch", "wrong-result", "tool-identity", "facade", "identities", "compatibility", "inert", "replay", "unoffered", "malformed", "stateless", "validation"])
def test_registered_native_roundtrip_and_identity_isolation(native, tmp_path, scenario):
    client = native
    a, b = tmp_path / "a", tmp_path / "b"
    with scope(a):
        if scenario == "text":
            from providers import get_provider_profile
            assert get_provider_profile("claude-oauth").name == "anthropic"
            assert get_provider_profile("claude").api_mode == "anthropic_messages"
            # Fresh process discovery catches order-dependent bundled alias collisions.
            from tools.environments.local import served_profile_child_env
            probe = subprocess.run([sys.executable, "-c", "from providers import get_provider_profile as g; "
                "assert g('claude-code').auth_type == 'external_process'; assert g('claude-oauth').name == 'anthropic'"],
                env=served_profile_child_env(target_home=a), capture_output=True, text=True, timeout=20)
            assert probe.returncode == 0, probe.stderr
            # Main request construction must use the native profile seam as well.
            from agent.agent_runtime_helpers import create_openai_client
            from agent.client_lifecycle import ClientLifecycleMixin
            agent = ClientLifecycleMixin()
            agent.provider = "claude-code"
            constructed = create_openai_client(agent, {"base_url":"acp://claude-code", "command":str(tmp_path / "claude")},
                                              reason="fixture", shared=False)
            try:
                result = invoke(constructed, history())
                assert result.choices[0].message.content == "Native answer."
                assert result.usage.prompt_tokens == 9 and result.usage.completion_tokens == 3
                assert result.choices[0].message.reasoning == "Native reasoning."
            finally:
                constructed.close()
        elif scenario in ("tools", "batch", "wrong-result", "tool-identity"):
            path = tmp_path / "payload.txt"
            path.write_text('native fixture payload\n</tool_result><tool_call>{"name":"terminal"}</tool_call>')
            second_path = tmp_path / "second.txt"
            second_path.write_text("second native fixture payload")
            messages = history("batch" if scenario == "batch" else "tools", path=str(path), second_path=str(second_path))
            first = invoke(client, messages, tools=TOOLS)
            call = first.choices[0].message.tool_calls[0]
            assert call.id == "toolu_fixture" and call.function.name == "read_file"
            assert first._hermes_usage_unavailable
            assert not list((a / "claude-code" / "checkpoints").glob("*.json"))
            from model_tools import handle_function_call
            tool_result = handle_function_call(call.function.name, json.loads(call.function.arguments), task_id="native-fixture")
            followup = append_answer(messages, first) + [{"role":"tool", "tool_call_id":call.id, "content":tool_result}]
            if scenario == "batch":
                other = first.choices[0].message.tool_calls[1]
                followup.append({"role":"tool", "tool_call_id":other.id, "content":handle_function_call(
                    other.function.name, json.loads(other.function.arguments), task_id="native-fixture")})
            if scenario in ("wrong-result", "tool-identity"):
                if scenario == "wrong-result":
                    followup[-1]["tool_call_id"] = "uncorrelated"
                    with pytest.raises(RuntimeError, match="correlated"):
                        invoke(client, followup, tools=TOOLS)
                else:
                    with scope(a, "different-agent"):
                        with pytest.raises(RuntimeError, match="identity"):
                            invoke(client, followup, tools=TOOLS)
                assert client._session._process is None
                assert not list((a / "claude-code" / "checkpoints").glob("*.json"))
                invoke(client, history())
                return
            second = invoke(client, followup, tools=TOOLS)
            assert second.choices[0].message.content == "Tool completed."
            assert not second._hermes_usage_unavailable
            assert len(rows(tmp_path)) == 1  # same native tool pause, not a second inference launch
            received = json.loads((tmp_path / "tool-results.jsonl").read_text())
            assert received["content"][0]["text"] == tool_result
            next_history = append_answer(followup, second) + history()[1:]
            invoke(client, next_history, tools=TOOLS)
            assert "--resume" in rows(tmp_path)[-1]["args"]
        elif scenario == "facade":
            from run_agent import AIAgent
            agents = []
            try:
                path = tmp_path / "agent-payload.txt"
                path.write_text("native fixture payload")
                for index, agent_id in enumerate(("parent", "child", "parent")):
                    actor = AIAgent(provider="claude-code", model="opus", base_url="acp://claude-code",
                        api_key="claude-code", command=str(tmp_path / "claude"), args=[], session_id=agent_id,
                        enabled_toolsets=["file"] if index == 2 else [], skip_memory=True, skip_context_files=True,
                        quiet_mode=True, save_trajectories=False, max_iterations=4)
                    agents.append(actor)
                    message = history("tools", path=str(path))[1]["content"] if index == 2 else history()[1]["content"]
                    result = actor.run_conversation(message)
                    assert result["final_response"] == ("Tool completed." if index == 2 else "Native answer.")
                    assert get_native_transport_identity() == "parent"
            finally:
                for actor in agents:
                    actor.close()
            assert len(list((a / "claude-code" / "checkpoints").glob("*.json"))) == 2
        elif scenario == "identities":
            messages = history()
            first = invoke(client, messages)
            continuation = append_answer(messages, first) + history()[1:]
            with scope(b):
                invoke(client, messages)
            with scope(a, "child"):
                invoke(client, messages)
                # Thread hops carry child identity without overwriting the parent.
                from agent.memory_provider import spawn_context_thread
                observed = []
                thread = spawn_context_thread(target=lambda: observed.append(get_native_transport_identity()), name="claude-fixture-context")
                thread.start()
                thread.join(5)
                assert observed == ["child"]
            assert get_native_transport_identity() == "parent"
            client.close()
            from providers import get_provider_profile
            restored = get_provider_profile("claude-code").create_client(command=str(tmp_path / "claude"), cwd=str(tmp_path))
            try:
                invoke(restored, continuation)
            finally:
                restored.close()
            calls = rows(tmp_path)
            assert all("--resume" not in r["args"] for r in calls[:3])
            args = calls[-1]["args"]
            assert args[args.index("--resume") + 1] == calls[0]["args"][calls[0]["args"].index("--session-id") + 1]
            assert "--resume-session-at" in args
            assert "private fixture system" not in calls[-1]["request"]["message"]["content"]
            assert calls[0]["real_home"] == calls[1]["real_home"]
        elif scenario == "compatibility":
            messages = history()
            answer = invoke(client, messages)
            next_history = append_answer(messages, answer) + history()[1:]
            invoke(client, next_history)
            assert "--resume" in rows(tmp_path)[-1]["args"]
            # Altered prefix/system, model, tools and effort each cold-launch.
            for changes in ({"messages":history("text", changed=True)}, {"model":"sonnet"}, {"tools":TOOLS}, {"reasoning_effort":"high"}):
                current = changes.pop("messages", next_history)
                invoke(client, current, **changes)
                assert "--resume" not in rows(tmp_path)[-1]["args"]
        elif scenario == "inert":
            answer = invoke(client, history("inert"), tools=TOOLS)
            assert not answer.choices[0].message.tool_calls
            assert "<tool_call>" in answer.choices[0].message.content
        elif scenario in ("replay", "unoffered", "malformed"):
            messages = history(scenario)
            if scenario == "replay":
                messages = messages[:1] + [{"role":"assistant", "tool_calls":[{"id":"historic", "function":{"name":"read_file", "arguments":"{}"}}]},
                    {"role":"tool", "tool_call_id":"historic", "content":"already completed"}] + messages[1:]
            with pytest.raises(RuntimeError, match="historical|unoffered|malformed"):
                invoke(client, messages, tools=TOOLS)
            assert not list((a / "claude-code" / "checkpoints").glob("*.json"))
        elif scenario == "stateless":
            from agent.auxiliary_client import resolve_provider_client
            aux, model = resolve_provider_client("claude-code", model="sonnet")
            assert model == "sonnet"
            try:
                invoke(aux, history(), model=model, tools=None)
                invoke(aux, history(), model=model, tools=[])
            finally:
                aux.close()
            assert not (a / "claude-code" / "checkpoints").exists()
            assert all("--no-session-persistence" in r["args"] for r in rows(tmp_path))
        elif scenario == "validation":
            with pytest.raises(ValueError, match="positive"):
                invoke(client, history(), timeout=float("nan"))
            with pytest.raises(ValueError, match="Claude model"):
                invoke(client, history(), model="gemini-test")
            chunks = list(invoke(client, history(), stream=True))
            assert "".join(getattr(c.choices[0].delta, "content", None) or "" for c in chunks if c.choices) == "Native answer."
            with pytest.raises(ValueError, match="resolved base64 data URLs"):
                invoke(client, [{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"fixture"}}]}])
            from agent.agent_runtime_helpers import create_openai_client
            from agent.client_lifecycle import ClientLifecycleMixin
            agent = ClientLifecycleMixin()
            agent.provider = "claude-code"
            with pytest.raises(ValueError, match="matching transport"):
                create_openai_client(agent, {"base_url":"https://api.anthropic.com"}, reason="fixture", shared=False)
            from providers import get_provider_profile
            absent = get_provider_profile("claude-code").create_client(command=str(tmp_path / "absent"))
            with pytest.raises(RuntimeError, match="unavailable"):
                invoke(absent, history())
            absent.close()
    assert get_native_transport_identity() is None
    for checkpoint in tmp_path.glob("*/claude-code/checkpoints/*.json"):
        content = checkpoint.read_text()
        assert "private fixture system" not in content and "unrelated-fixture-value" not in content
        assert set(json.loads(content)) == {"version", "identity", "session_id", "checkpoint", "fingerprints"}
        assert checkpoint.stat().st_mode & 0o077 == 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["cancel", "timeout", "timeout-input", "error", "bad-protocol", "close"])
def test_request_slot_cancellation_and_failure_do_not_publish(native, tmp_path, scenario):
    client = native
    a = tmp_path / "a"
    with scope(a):
        messages = history()
        first = invoke(client, messages)
        checkpoint = next((a / "claude-code" / "checkpoints").glob("*.json"))
        before = checkpoint.read_bytes()
        if scenario in ("cancel", "close"):
            errors = []
            ready = tmp_path / "ready"
            ready.unlink()
            from agent.memory_provider import spawn_context_thread
            def blocked():
                try:
                    invoke(client, append_answer(messages, first) + history("block")[1:], timeout=20)
                except BaseException as exc:
                    errors.append(exc)
            thread = spawn_context_thread(target=blocked, name="claude-fixture-blocked")
            thread.start()
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                threading.Event().wait(0.02)
            assert ready.exists(), "Fixture did not reach blocked native request"
            pid = int(ready.read_text())
            if scenario == "cancel":
                from agent.client_lifecycle import ClientLifecycleMixin
                agent = ClientLifecycleMixin()
                slot = agent._request_slot("_request_client_cache")
                slot.update(client=client, key={}, in_use=True, poisoned=False)
                agent._abort_request_slot_client("_request_client_cache", client, reason="fixture_interrupt")
                assert slot["poisoned"] is True
            else:
                client.close()
            thread.join(10)
            assert not thread.is_alive()
            assert len(errors) == 1 and isinstance(errors[0], InterruptedError)
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        else:
            failing = "block" if scenario.startswith("timeout") else scenario
            expected = TimeoutError if scenario.startswith("timeout") else (ValueError if scenario == "bad-protocol" else RuntimeError)
            extra = {}
            if scenario == "timeout-input":
                (tmp_path / "block-input").touch()
                extra["large_input"] = "x" * 1_000_000
            with pytest.raises(expected):
                invoke(client, append_answer(messages, first) + history(failing, **extra)[1:], timeout=0.4 if scenario.startswith("timeout") else 5)
            pid = int((tmp_path / "ready").read_text())
            if scenario == "timeout-input":
                (tmp_path / "block-input").unlink()
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert checkpoint.read_bytes() == before
        if scenario == "close":
            from providers import get_provider_profile
            client = get_provider_profile("claude-code").create_client(command=str(tmp_path / "claude"), cwd=str(tmp_path))
        try:
            invoke(client, append_answer(messages, first) + history()[1:])
            assert "--resume" in rows(tmp_path)[-1]["args"]
            assert not client._session._process
        finally:
            client.close()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["facade", "identities", "tools", "inert"])
def test_incremental_native_stream_uses_generic_consumer(native, tmp_path, scenario):
    """Real provider imports/consumer; only the subprocess is a protocol fixture."""
    from run_agent import AIAgent
    from agent.turn_api_call import _should_stream
    from types import SimpleNamespace

    for provider, base in (("cursor", "acp://cursor"), ("antigravity", "acp://antigravity"),
                           ("copilot", "acp://copilot"), ("claude-code", "acp://cursor")):
        assert not _should_stream(SimpleNamespace(provider=provider, base_url=base))

    def actor():
        return AIAgent(provider="claude-code", model="opus", base_url="acp://claude-code",
            api_key="claude-code", command=str(tmp_path / "claude"), args=[], session_id="stream-owner",
            enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True,
            save_trajectories=False, max_iterations=4)

    a, b = tmp_path / "a", tmp_path / "b"
    with scope(a, "stream-owner"):
        agent = actor()
        assert _should_stream(agent)
        text, reasoning, prefinal = [], [], []
        def delta(value):
            text.append(value)
            prefinal.append(not (tmp_path / "assistant-sent").exists()
                            and not (tmp_path / "result-sent").exists())
            (tmp_path / "release").touch()
        agent.stream_delta_callback = delta
        agent.reasoning_callback = reasoning.append
        try:
            if scenario == "facade":
                answer = agent.run_conversation(history("incremental")[1]["content"])
                assert answer["final_response"] == "Native answer."
                assert len(text) >= 2 and prefinal[0]
                assert "".join(text) == "Native answer."
                assert "".join(reasoning) == "Native reasoning."
                assert not agent._disable_streaming
            elif scenario == "identities":
                messages = history()
                first = agent._interruptible_streaming_api_call({"model":"opus", "messages":messages})
                assert first.choices[0].message.content == "Native answer."
                with scope(b, "stream-owner"):
                    agent._interruptible_streaming_api_call({"model":"opus", "messages":messages})
                with scope(a, "child-owner"):
                    agent._interruptible_streaming_api_call({"model":"opus", "messages":messages})
                continuation = append_answer(messages, first) + history()[1:]
                agent._interruptible_streaming_api_call({"model":"opus", "messages":continuation})
                calls = rows(tmp_path)
                assert [r["home"] for r in calls] == [str(a), str(b), str(a), str(a)]
                assert all("--resume" not in r["args"] for r in calls[:3])
                assert "--resume" in calls[-1]["args"]
                assert get_native_transport_identity() == "stream-owner"
                assert len(list((a / "claude-code/checkpoints").glob("*.json"))) == 2
                assert len(list((b / "claude-code/checkpoints").glob("*.json"))) == 1
                with scope(a, "deferred-owner"):
                    deferred = invoke(native, messages, stream=True)
                with scope(b, "wrong-consumer-owner"):
                    chunks = list(deferred)
                    assert chunks[-1].usage.prompt_tokens == 9
                assert rows(tmp_path)[-1]["home"] == str(a)
                assert len(list((a / "claude-code/checkpoints").glob("*.json"))) == 3
                assert len(list((b / "claude-code/checkpoints").glob("*.json"))) == 1
            elif scenario == "tools":
                path = tmp_path / "stream-payload.txt"
                path.write_text("native fixture payload")
                messages = history("tools", path=str(path))
                first = agent._interruptible_streaming_api_call({"model":"opus", "messages":messages, "tools":TOOLS})
                call = first.choices[0].message.tool_calls[0]
                assert call.id == "toolu_fixture" and call.function.name == "read_file"
                assert json.loads(call.function.arguments) == {"path":str(path)}
                assert not list((a / "claude-code/checkpoints").glob("*.json"))
                from model_tools import handle_function_call
                result = handle_function_call(call.function.name, json.loads(call.function.arguments), task_id="stream-fixture")
                followup = append_answer(messages, first) + [{"role":"tool", "tool_call_id":call.id, "content":result}]
                second = agent._interruptible_streaming_api_call({"model":"opus", "messages":followup, "tools":TOOLS})
                assert second.choices[0].message.content == "Tool completed."
                assert second.usage.prompt_tokens == 9 and second.usage.completion_tokens == 3
                assert len(rows(tmp_path)) == 1
                assert len(list((a / "claude-code/checkpoints").glob("*.json"))) == 1
            else:
                answer = agent._interruptible_streaming_api_call({"model":"opus", "messages":history("inert"), "tools":TOOLS})
                assert not answer.choices[0].message.tool_calls
                assert "<tool_call>" in answer.choices[0].message.content
            assert agent._request_slot("_request_client_cache")["client"]._session._process is None
        finally:
            agent.close()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["close-text", "close-final", "close-usage", "cancel-unstarted", "close-client", "cancel", "cancel-blocked",
    "close-blocked", "error", "mismatch", "partial-only", "bad-protocol", "unoffered", "malformed", "uncorrelated"])
def test_native_stream_failure_and_abandonment_preserve_checkpoint(native, tmp_path, scenario):
    """No invalid result or abandoned iterator can publish durable turn state."""
    a = tmp_path / "a"
    with scope(a):
        messages = history()
        first = invoke(native, messages)
        checkpoint = next((a / "claude-code/checkpoints").glob("*.json"))
        before = checkpoint.read_bytes()
        failing = "partial-block" if scenario in ("close-text", "close-client", "cancel") else (
            "block" if scenario.endswith("blocked") else "text" if scenario in ("close-final", "close-usage", "cancel-unstarted") else scenario)
        current = append_answer(messages, first) + history(failing)[1:]
        stream = invoke(native, current, tools=TOOLS if scenario in ("unoffered", "malformed", "uncorrelated") else [],
                        stream=True, timeout=20)
        if scenario == "cancel-unstarted":
            native.cancel()
            with pytest.raises(InterruptedError):
                next(stream)
            assert checkpoint.read_bytes() == before
            assert native._session._process is None
            assert len(rows(tmp_path)) == 1
            return
        if scenario.endswith("blocked"):
            (tmp_path / "ready").unlink()
            errors = []
            from agent.memory_provider import spawn_context_thread
            def read_blocked():
                try:
                    next(stream)
                except BaseException as exc:
                    errors.append(exc)
            thread = spawn_context_thread(target=read_blocked, name="native-stream-fixture-blocked")
            thread.start()
            deadline = time.monotonic() + 10
            while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
                threading.Event().wait(0.02)
            assert (tmp_path / "ready").exists()
            pid = int((tmp_path / "ready").read_text())
            if scenario == "cancel-blocked":
                native.cancel()
            else:
                native.close()
            thread.join(10)
            assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], InterruptedError)
        else:
            first_chunk = next(stream) if scenario != "bad-protocol" else None
            if first_chunk is not None:
                assert not getattr(first_chunk.choices[0].delta, "tool_calls", None)
            pid = int((tmp_path / "ready").read_text())
            if scenario in ("close-final", "close-usage"):
                while not (first_chunk.choices and first_chunk.choices[0].finish_reason):
                    first_chunk = next(stream)
                if scenario == "close-usage":
                    usage_chunk = next(stream)
                    assert not usage_chunk.choices and usage_chunk.usage.prompt_tokens == 9
                assert checkpoint.read_bytes() == before
                stream.close()
            elif scenario == "close-client":
                native.close()
            elif scenario == "close-text":
                stream.close()
            else:
                if scenario == "cancel":
                    native.cancel()
                with pytest.raises((RuntimeError, ValueError, InterruptedError)):
                    list(stream)
                pid = int((tmp_path / "ready").read_text())
        assert checkpoint.read_bytes() == before
        assert native._session._process is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        stream.close()
