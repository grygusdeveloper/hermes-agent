"""Cold Claude native stream-json transport with a bounded Hermes MCP tool pause.

Native argv, real-system-slot, exact checkpoint resume and credential boundary
are semantically ported from clean fork ClaudeCodeSession (9992a59bb6, introduced
2a8696c42e). SDK MCP control envelopes are Claude's native stdin/stdout protocol,
not ACP. No HTTP transport or credential reader is present.
"""
from __future__ import annotations

import contextvars
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from agent.claude_code_checkpoint import Checkpoint
from agent.claude_code_events import NativeTurn
from agent.claude_code_protocol import (
    NAMESPACE, canonical,
    control_response, digest, message_fingerprints, system_and_transcript,
    final_chunks,
)
from hermes_constants import get_hermes_home, get_real_home
from hermes_platform.resolver.core import locate_command
from tools.environments.local import served_profile_child_env


class NativeStream:
    """Pull-based stream; close cancels I/O before waiting for the owning reader.

    No native producer thread or unbounded queue: Hermes's existing context-aware
    streaming worker drives next(). Capture scope even if consumption moves threads.
    """
    def __init__(self, session, iterator):
        self.session, self.iterator = session, iterator
        self.context = contextvars.copy_context()
        self.lock = threading.Lock()
        self.finished = self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.closed or self.finished:
                raise StopIteration
            try:
                return self.context.run(next, self.iterator)
            except BaseException:
                self.finished = True
                raise

    def close(self):
        if self.finished or self.closed:
            return
        self.closed = True
        self.session.cancel()
        with self.lock:
            self.context.run(self.iterator.close)
            self.finished = True


class ClaudeCodeSession:
    def __init__(self, command: str, cwd: str | None):
        self.command = command
        self.cwd = cwd
        self._serial = threading.Lock()
        self._guard = threading.RLock()
        self._reap_guard = threading.Lock()
        self._reaped_process = None
        self._epoch = 0
        self._closed = False
        self._process = None
        self._checkpoint = None
        self._selector = None
        self._temp = None
        self._pending: dict | None = None
        self._buffer = b""

    def _reap(self, process) -> None:
        # Always kill the group even if the leader exited: native connectors
        # must not survive cancellation. No stranger-thread FD close.
        # cancel() and the reader's finally can race. Reap each owned group
        # once, never signal a now-recycled PID after another thread waited it.
        with self._reap_guard:
            if process is self._reaped_process:
                return
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            self._reaped_process = process

    def cancel(self) -> None:
        with self._guard:
            self._epoch += 1
            process = self._process
        if process is not None:
            self._reap(process)

    def close(self) -> None:
        with self._guard:
            self._closed = True
        self.cancel()
        with self._serial:
            self._cleanup()

    def _cleanup(self) -> None:
        process = self._process
        if process is not None:
            self._reap(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                assert stream is not None
                stream.close()
        if self._selector is not None:
            self._selector.close()
        if self._checkpoint is not None:
            self._checkpoint.close()
        if self._temp is not None:
            self._temp.cleanup()
        with self._guard:
            self._process = self._selector = self._checkpoint = self._temp = self._pending = None
        self._buffer = b""

    def _check(self, epoch: int, deadline: float) -> None:
        if self._closed or epoch != self._epoch:
            raise InterruptedError("Claude native request cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Claude native request timed out")

    def _send(self, payload: dict) -> None:
        process = self._process
        assert process is not None and process.stdin is not None
        selector = self._selector
        assert selector is not None
        remaining = memoryview((canonical(payload) + "\n").encode())
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        try:
            while remaining:
                self._check(self._io_epoch, self._deadline)
                for key, _ in selector.select(min(0.2, max(0, self._deadline - time.monotonic()))):
                    if key.data == "stdin":
                        remaining = remaining[os.write(key.fd, remaining[:65536]):]
                    else:
                        self._read_ready(key, self._io_epoch, self._deadline)
        finally:
            selector.unregister(process.stdin)

    def _read_ready(self, key, epoch: int, deadline: float) -> None:
        selector = self._selector
        assert selector is not None
        data = os.read(key.fd, 65536)
        if not data:
            selector.unregister(key.fileobj)
            if key.data == "stdout":
                self._check(epoch, deadline)
                raise RuntimeError("Claude CLI exited before a successful native result")
        elif key.data == "stdout":
            self._buffer += data
        # stderr is drained without persisting/echoing prompts or credentials.

    def _event(self, epoch: int, deadline: float) -> dict:
        import json
        selector = self._selector
        assert selector is not None
        while True:
            self._check(epoch, deadline)
            if b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                value = json.loads(raw)
                if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                    raise RuntimeError("Unsupported Claude native event")
                return value
            if len(self._buffer) > 16 * 1024 * 1024:
                raise RuntimeError("Claude native event exceeded size limit")
            for key, _ in selector.select(min(0.2, max(0, deadline - time.monotonic()))):
                self._read_ready(key, epoch, deadline)

    def _mcp_reply(self, event: dict, result: dict) -> None:
        message = event["request"]["message"]
        rpc = {"jsonrpc": "2.0", "result": result}
        if "id" in message:
            rpc["id"] = message["id"]
        self._send(control_response(event["request_id"], {"mcp_response": rpc}))

    def _control(self, event: dict, specs: list[dict], calls: list[dict]) -> dict | None:
        request = event.get("request", {})
        subtype = request.get("subtype")
        if subtype == "can_use_tool":
            name = request.get("tool_name", "")
            if not name.startswith(NAMESPACE) or name.removeprefix(NAMESPACE) not in {s["name"] for s in specs}:
                raise RuntimeError("Claude CLI requested permission for an unoffered tool")
            self._send(control_response(event["request_id"], {
                "behavior": "allow", "updatedInput": request.get("input", {})}))
            return None
        if subtype != "mcp_message" or request.get("server_name") != "hermes":
            raise RuntimeError("Unsupported Claude native control request")
        message = request.get("message", {})
        method = message.get("method")
        if method == "tools/call":
            params = message.get("params", {})
            matches = [c for c in calls if c["function"]["name"] == params.get("name")
                       and c["function"]["arguments"] == canonical(params.get("arguments", {}))]
            if len(matches) != 1:
                raise RuntimeError("Claude MCP call is not correlated to one native tool_use")
            return {"event": event, "call": matches[0]}
        replies = {
            "initialize": {"protocolVersion": message.get("params", {}).get("protocolVersion", "2024-11-05"),
                           "capabilities": {"tools": {}}, "serverInfo": {"name": "hermes", "version": "1"}},
            "tools/list": {"tools": specs}, "notifications/initialized": {}, "ping": {},
        }
        if method not in replies:
            raise RuntimeError("Unsupported Claude Hermes MCP method")
        self._mcp_reply(event, replies[method])
        return None

    def _launch(self, home: Path, system: str, specs: list[dict], model: str, effort: str | None,
                state: dict | None, epoch: int) -> None:
        command = locate_command(self.command).command
        if not command:
            raise RuntimeError("Claude native CLI is unavailable; no HTTP fallback is permitted")
        self._temp = tempfile.TemporaryDirectory(prefix="claude-native-", dir=home)
        prompt_file = Path(self._temp.name) / "system.txt"
        prompt_file.write_text(system)
        prompt_file.chmod(0o600)
        argv = [*command, "-p", "--model", model, "--input-format", "stream-json",
                "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--system-prompt-file", str(prompt_file),
                "--tools", "", "--strict-mcp-config", "--mcp-config",
                canonical({"mcpServers": {"hermes": {"type": "sdk", "name": "hermes"}}} if specs else {"mcpServers": {}}),
                "--permission-prompt-tool", "stdio", "--setting-sources", "", "--disable-slash-commands"]
        if effort:
            argv += ["--effort", effort]
        if state:
            argv += ["--resume", state["session_id"], "--resume-session-at", state["checkpoint"]]
        else:
            argv += ["--session-id", str(uuid.uuid4())]
            if self._checkpoint is None:
                argv += ["--no-session-persistence"]
        env = served_profile_child_env(target_home=home, inherit_credentials=False)
        if real_home := get_real_home(env):
            env["HOME"] = real_home  # CLI keeps its own OAuth/keychain, never profile sandbox HOME.
        for key in list(env):
            if key.upper().startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) or key in (
                "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_API_KEY", "CLAUDE_CONFIG_DIR", "CLAUDECODE"):
                env.pop(key, None)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        with self._guard:
            self._check(epoch, float("inf"))
            self._process = subprocess.Popen(argv, cwd=self.cwd or str(home), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self._selector = selectors.DefaultSelector()
        assert self._process.stdin is not None and self._process.stdout is not None and self._process.stderr is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        self._selector.register(self._process.stdout, selectors.EVENT_READ, "stdout")
        self._selector.register(self._process.stderr, selectors.EVENT_READ, "stderr")
        self._send({"type": "control_request", "request_id": "hermes-init", "request": {"subtype": "initialize", "hooks": {}}})

    def run(self, **kwargs):
        iterator = self._iterate(**kwargs, streaming=False)
        while True:
            try:
                next(iterator)
            except StopIteration as end:
                return end.value

    def stream(self, **kwargs):
        with self._guard:
            epoch = self._epoch
        return NativeStream(self, self._iterate(**kwargs, streaming=True, request_epoch=epoch))

    def _iterate(self, *, model: str, messages: list[dict], specs: list[dict], owner: str | None,
                 timeout: float, effort: str | None = None, required: bool = False, streaming: bool,
                 request_epoch: int | None = None):
        with self._serial:
            with self._guard:
                epoch = self._epoch if request_epoch is None else request_epoch
            deadline = time.monotonic() + timeout
            parked = False
            self._io_epoch, self._deadline = epoch, deadline
            try:
                self._check(epoch, deadline)
                home = get_hermes_home().resolve()
                home.mkdir(parents=True, exist_ok=True)
                system, transcript = system_and_transcript(messages)
                identity = digest([str(home), owner, self.command, self.cwd or str(home), model, effort, system, specs])
                fingerprints = message_fingerprints(messages)
                turn = NativeTurn(self, model=model, messages=messages, specs=specs, identity=identity,
                                  epoch=epoch, deadline=deadline, transcript=transcript, required=required,
                                  streaming=streaming)
                if self._pending:
                    pending = self._pending
                    self._check(pending["epoch"], deadline)
                    if identity != pending["identity"] or fingerprints[:len(pending["fingerprints"])] != pending["fingerprints"]:
                        raise RuntimeError("Claude native pending tool identity or message prefix changed")
                    tail = messages[len(pending["fingerprints"]):]
                    results = {row.get("tool_call_id"): row for row in tail if row.get("role") == "tool"}
                    if len(results) != len(tail) or set(results) != {call["id"] for call in pending["calls"]}:
                        raise RuntimeError("Claude native requires the correlated Hermes tool_result")
                    turn.calls, turn.tool_results = list(pending["calls"]), results
                    turn.blocks = dict(pending["blocks"])
                    turn.supply_result(pending["event"], pending["call"])
                    turn.session_id = pending["session_id"]
                    self._pending = None
                    turn.initialized = True
                else:
                    self._checkpoint = Checkpoint(home, owner) if owner else None
                    state = self._checkpoint.load() if self._checkpoint else None
                    if state and (state["identity"] != identity or fingerprints[:len(state["fingerprints"])] != state["fingerprints"]):
                        state = None
                    if state:
                        new = messages[len(state["fingerprints"]):]
                        if not new:
                            raise RuntimeError("Claude native request does not advance the published prefix")
                        # CLI already holds all assistant entries through the exact
                        # checkpoint. Only new records go on stdin.
                        _, transcript = system_and_transcript(new)
                        turn.transcript = transcript
                    self._launch(home, system, specs, model, effort, state, epoch)
                while True:
                    event = self._event(epoch, deadline)
                    result = turn.dispatch(event)
                    yield from turn.chunks
                    turn.chunks.clear()
                    if result is not None:
                        if streaming:
                            yield from final_chunks(result)
                        # Only full consumption may retain the MCP pause or publish
                        # durable state. Generator.close/error/cancel takes cleanup.
                        turn.commit()
                        parked = turn.parked
                        return result
            finally:
                if not parked:
                    self._cleanup()

    @staticmethod
    def _compatible_model(requested: str, actual: str) -> bool:
        if requested == actual:
            return True
        return requested in ("opus", "sonnet", "haiku", "fable") and actual.startswith("claude-" + requested + "-")
