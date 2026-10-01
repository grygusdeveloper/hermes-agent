"""Cold Claude native stream-json transport with a bounded Hermes MCP tool pause.

Native argv, real-system-slot, exact checkpoint resume and credential boundary
are semantically ported from clean fork ClaudeCodeSession (9992a59bb6, introduced
2a8696c42e). SDK MCP control envelopes are Claude's native stdin/stdout protocol,
not ACP. No HTTP transport or credential reader is present.
"""
from __future__ import annotations

import contextvars
import math
import errno
import logging
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from agent.claude_code_checkpoint import Checkpoint
from agent.claude_code_errors import (
    HARD_LIMIT_TEXT_RE, ClaudeCodeLaunchError, ClaudeCodeSessionExpired, ClaudeCodeSoftLimitNotice,
    CliFlagRejected, is_expired_session_error, soft_limit_exhausted,
)
from agent.claude_code_events import NativeTurn
from agent.claude_code_limits import (
    block_usage_limit, limit_window, mark_limit_warnings_delivered, overage_usable, record_rate_limit,
    usage_limit_block, usage_limit_error,
)
from agent.claude_code_protocol import (
    NAMESPACE, canonical,
    control_response, digest, message_fingerprints, system_and_transcript,
    final_chunks,
)
from hermes_constants import get_hermes_home, get_real_home
from hermes_platform.resolver.core import locate_command
from tools.environments.local import served_profile_child_env

logger = logging.getLogger(__name__)

# ``--thinking-display`` is a hidden CLI flag (2.1.276+): Opus 5 and newer default to
# omitted thinking, so without it every thinking block reaches Hermes empty. Override
# with HERMES_CLAUDE_CODE_THINKING_DISPLAY (``off`` sends no flag); turned off for the
# process if a CLI rejects it (46d684c584).
_THINKING_DISPLAY_CHOICES = frozenset({"summarized", "omitted"})
_thinking_display_supported = True
# Child environment defaults; a value already set in Hermes's own environment wins.
_CLI_ENV_DEFAULTS = {
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_TELEMETRY": "1",
    # Hermes owns compaction: Claude Code compacting its copy would silently
    # desync it from the transcript Hermes keeps resuming.
    "DISABLE_AUTO_COMPACT": "1",
    # Pinned 1-hour prompt cache: chat replies often come more than 5 minutes apart.
    "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h",
    # One reply is one step of a chat turn, never a 64k-token document.
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16000",
}
# Soft usage notices are retried in-session, resumed at the same checkpoint so the
# rejected attempt is dropped (9112f82f46, 232446c398).
_SOFT_LIMIT_ATTEMPTS = 3
_SOFT_LIMIT_BASE_DELAY = 2.0
# The same fallback notice is shown at most this often per session: a flagged
# conversation falls back again on every call of a tool loop (46eb9e7345).
_NOTICE_REPEAT_SECONDS = 600.0
_MAX_PENDING_NOTICES = 8
_STDERR_TAIL_BYTES = 8192
# Optional flags a CLI may reject before reading the request (each is then turned
# off for the process and the request respawned without it).
_OPTIONAL_FLAG_RESPAWNS = 2
# Popen failures that repeat until the installation or configuration is fixed.
# Resource errors and ETXTBSY (the auto-updater replacing the binary) stay retryable.
_LAUNCH_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.EPERM, errno.ENOEXEC,
                            errno.ELOOP, errno.ENAMETOOLONG, errno.EISDIR})


def _thinking_display() -> str | None:
    if not _thinking_display_supported:
        return None
    value = os.getenv("HERMES_CLAUDE_CODE_THINKING_DISPLAY", "").strip().lower() or "summarized"
    return value if value in _THINKING_DISPLAY_CHOICES else None


def _cli_env_defaults(env: dict[str, str]) -> None:
    for key, value in _CLI_ENV_DEFAULTS.items():
        env.setdefault(key, os.environ.get(key, value))


_KEEPALIVE_DEFAULT_SECONDS = 300.0
_MAX_WARM_DEFAULT = 3
_WARM_LOCK = threading.Lock()
# Idle warm sessions, oldest first (bounded by HERMES_CLAUDE_CODE_MAX_WARM).
_WARM_SESSIONS: "OrderedDict[int, ClaudeCodeSession]" = OrderedDict()


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _keepalive_seconds() -> float:
    return _env_number("HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS", _KEEPALIVE_DEFAULT_SECONDS)


def _max_warm() -> int:
    return int(_env_number("HERMES_CLAUDE_CODE_MAX_WARM", _MAX_WARM_DEFAULT))


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
        self._stderr = b""
        self._resuming = False
        # Latest rate_limit_info this session saw, and notices (plan warnings,
        # CLI model fallbacks) waiting for a reply that shows them.
        self._last_rate_limit: dict = {}
        self._pending_notices: list[tuple[str, tuple]] = []
        self._shown_notices: dict[str, float] = {}
        self._notice_lock = threading.Lock()
        # Idle warm process: ((identity, session_id, checkpoint, fingerprints), until).
        self._warm = None
        self._warm_timer = None

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
        self._take_warm()
        self._discard_process()
        if self._checkpoint is not None:
            self._checkpoint.close()
        with self._guard:
            self._checkpoint = self._pending = None

    def _discard_process(self) -> None:
        """Reap the CLI and its I/O; the checkpoint lock stays (a retry relaunches)."""
        process = self._process
        if process is not None:
            self._reap(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                assert stream is not None
                stream.close()
        if self._selector is not None:
            self._selector.close()
        if self._temp is not None:
            self._temp.cleanup()
        with self._guard:
            self._process = self._selector = self._temp = None
        self._buffer = self._stderr = b""

    # ── Warm processes across user turns ─────────────────────────────
    # A durable conversation's CLI stays up for HERMES_CLAUDE_CODE_KEEPALIVE_SECONDS
    # (default 300) after a published answer, so the next user turn skips the
    # spawn and --resume (2338745987, f8a124c66f). At most HERMES_CLAUDE_CODE_MAX_WARM
    # (default 3) idle processes exist; the checkpoint lock is not held while idle.

    def _warm_record(self, turn):
        seconds = _keepalive_seconds()
        process = self._process
        if (seconds <= 0 or _max_warm() <= 0 or turn.published is None or self._checkpoint is None
                or process is None or process.poll() is not None):
            return None
        return turn.published, time.monotonic() + seconds, seconds

    def _park_warm(self, published, until: float, seconds: float) -> None:
        if self._checkpoint is not None:
            self._checkpoint.close()
        timer = threading.Timer(seconds + 0.05, self._expire_warm)
        timer.daemon = True
        with self._guard:
            self._checkpoint = self._pending = None
            self._warm, self._warm_timer = (published, until), timer
        timer.start()
        evicted = []
        with _WARM_LOCK:
            _WARM_SESSIONS.pop(id(self), None)
            _WARM_SESSIONS[id(self)] = self
            while len(_WARM_SESSIONS) > _max_warm():
                evicted.append(_WARM_SESSIONS.popitem(last=False)[1])
        for other in evicted:
            other._expire_warm(force=True)

    def _take_warm(self):
        """The idle process's published state, unparked (None when absent or expired)."""
        with self._guard:
            warm, timer = self._warm, self._warm_timer
            self._warm = self._warm_timer = None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        with _WARM_LOCK:
            if _WARM_SESSIONS.get(id(self)) is self:
                del _WARM_SESSIONS[id(self)]
        if warm is None:
            return None
        if time.monotonic() >= warm[1]:
            self._discard_process()
            return None
        return warm[0]

    def _expire_warm(self, force: bool = False) -> None:
        # A request that holds the session takes the warm process itself.
        if not self._serial.acquire(blocking=False):
            return
        try:
            warm = self._warm
            if warm is not None and (force or time.monotonic() >= warm[1]):
                self._take_warm()
                self._discard_process()
        finally:
            self._serial.release()

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
                        try:
                            remaining = remaining[os.write(key.fd, remaining[:65536]):]
                        except BrokenPipeError:
                            # The CLI exited before reading the request (e.g. an
                            # argv it rejected); its stderr says why.
                            self._check(self._io_epoch, self._deadline)
                            raise self._exit_error() from None
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
                raise self._exit_error()
        elif key.data == "stdout":
            self._buffer += data
        else:
            # A bounded stderr tail, kept only to classify an early exit (rejected
            # flag, unknown session); never persisted or echoed into errors.
            self._stderr = (self._stderr + data)[-_STDERR_TAIL_BYTES:]

    def _exit_error(self) -> BaseException:
        """Why the CLI exited before a result, from its own stderr."""
        global _thinking_display_supported
        process, until = self._process, time.monotonic() + 1.0
        stream = process.stderr if process is not None else None
        if stream is not None and not stream.closed:
            # Its last stderr words may still be in the pipe after stdout closed.
            with selectors.DefaultSelector() as drain:
                drain.register(stream, selectors.EVENT_READ)
                while (left := until - time.monotonic()) > 0 and drain.select(left):
                    data = os.read(stream.fileno(), 65536)
                    if not data:
                        break
                    self._stderr = (self._stderr + data)[-_STDERR_TAIL_BYTES:]
        stderr = self._stderr.decode("utf-8", "replace").lower()
        if "--thinking-display" in stderr and ("unknown option" in stderr or "allowed choices" in stderr):
            _thinking_display_supported = False
            logger.warning("Claude Code CLI rejected --thinking-display; Claude's reasoning stays hidden from now on")
            return CliFlagRejected("Claude Code CLI rejected --thinking-display")
        if self._resuming and is_expired_session_error(stderr):
            return ClaudeCodeSessionExpired("Claude Code could not resume the saved session (no conversation found)")
        return RuntimeError("Claude CLI exited before a successful native result")

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
                state: dict | None, epoch: int, thinking: str | None = None) -> None:
        command = locate_command(self.command).command
        if not command:
            raise ClaudeCodeLaunchError(
                f"Claude Code CLI not found ({self.command!r}): the native CLI is unavailable; "
                "no HTTP fallback is permitted")
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
        if display := _thinking_display():
            argv += ["--thinking-display", display]
        if effort:
            argv += ["--effort", effort]
        if thinking:
            argv += ["--thinking", thinking]
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
        _cli_env_defaults(env)
        cwd = self.cwd or str(home)
        self._resuming = bool(state)
        with self._guard:
            self._check(epoch, float("inf"))
            try:
                self._process = subprocess.Popen(argv, cwd=cwd, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            except OSError as exc:
                if not (isinstance(exc, (FileNotFoundError, NotADirectoryError, PermissionError, IsADirectoryError))
                        or exc.errno in _LAUNCH_ERRNOS):
                    raise
                raise ClaudeCodeLaunchError(
                    f"Claude Code CLI not launchable: {command[0]!r} in working directory {cwd!r} "
                    f"({exc.strerror or exc})") from None
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
                 request_epoch: int | None = None, thinking: str | None = None):
        with self._serial:
            with self._guard:
                epoch = self._epoch if request_epoch is None else request_epoch
            deadline = time.monotonic() + timeout
            parked = False
            keep = None
            self._io_epoch, self._deadline = epoch, deadline
            try:
                self._check(epoch, deadline)
                limited = usage_limit_block(model)
                if limited is not None:
                    # At a subscription limit for this model until its reset: the
                    # CLI would only print the same notice again.
                    logger.info("Claude Code not started: %s", limited)
                    raise limited
                home = get_hermes_home().resolve()
                home.mkdir(parents=True, exist_ok=True)
                system, transcript = system_and_transcript(messages, elide_old_calls=True)
                # Thinking joins the identity only when set, so pre-existing checkpoints keep matching.
                identity = digest([str(home), owner, self.command, self.cwd or str(home), model, effort, system, specs]
                                  + ([thinking] if thinking else []))
                fingerprints = message_fingerprints(messages)

                def new_turn(transcript, mode):
                    return NativeTurn(self, model=model, messages=messages, specs=specs, identity=identity,
                                      epoch=epoch, deadline=deadline, transcript=transcript, required=required,
                                      streaming=streaming, mode=mode)

                if self._pending:
                    pending = self._pending
                    self._check(pending["epoch"], deadline)
                    if identity != pending["identity"] or fingerprints[:len(pending["fingerprints"])] != pending["fingerprints"]:
                        raise RuntimeError("Claude native pending tool identity or message prefix changed")
                    tail = messages[len(pending["fingerprints"]):]
                    results = {row.get("tool_call_id"): row for row in tail if row.get("role") == "tool"}
                    if len(results) != len(tail) or set(results) != {call["id"] for call in pending["calls"]}:
                        raise RuntimeError("Claude native requires the correlated Hermes tool_result")
                    turn = new_turn("", "tool")
                    turn.calls, turn.tool_results = list(pending["calls"]), results
                    turn.blocks = dict(pending["blocks"])
                    turn.supply_result(pending["event"], pending["call"])
                    turn.session_id = pending["session_id"]
                    self._pending = None
                    turn.initialized = True
                    try:
                        result = yield from self._drive(turn, claims=owner is not None)
                    except ClaudeCodeSoftLimitNotice as exc:
                        # Mid tool loop the CLI holds the call's result: never relaunch
                        # here, but a hard limit still fails fast until its reset.
                        self._raise_if_usage_limited(exc.detail or str(exc), model, cause=exc)
                        raise
                else:
                    warm = self._take_warm()
                    self._checkpoint = Checkpoint(home, owner) if owner else None
                    state = self._checkpoint.load() if self._checkpoint else None
                    if state and (state["identity"] != identity or fingerprints[:len(state["fingerprints"])] != state["fingerprints"]):
                        state = None
                    # The parked process holds exactly the published conversation:
                    # anything else (another process advanced it, a rewind, /new, an
                    # identity change) resumes cold instead.
                    reuse = bool(warm and state and self._process is not None and self._process.poll() is None
                                 and warm == (identity, state["session_id"], state["checkpoint"], state["fingerprints"]))
                    if warm and not reuse:
                        self._discard_process()
                    soft_limits = flag_respawns = 0
                    while True:
                        if state:
                            new = messages[len(state["fingerprints"]):]
                            if not new:
                                raise RuntimeError("Claude native request does not advance the published prefix")
                            # CLI already holds all assistant entries through the exact
                            # checkpoint. Only new records go on stdin.
                            turn = new_turn(system_and_transcript(new)[1], "resume")
                        else:
                            turn = new_turn(transcript, "cold")
                        try:
                            if reuse:
                                reuse = False
                                turn.mode, turn.initialized, turn.session_id = "warm", True, state["session_id"]
                                self._send({"type": "user", "message": {"role": "user", "content": turn.transcript}})
                            else:
                                self._launch(home, system, specs, model, effort, state, epoch, thinking)
                            result = yield from self._drive(turn, claims=owner is not None)
                            break
                        except CliFlagRejected:
                            # The flag is now off for the process (see _exit_error): respawn
                            # without it, once per optional flag a CLI may reject.
                            flag_respawns += 1
                            if turn.emitted or flag_respawns > _OPTIONAL_FLAG_RESPAWNS:
                                raise
                        except ClaudeCodeSessionExpired:
                            # Expired server session or unknown checkpoint: forget it and
                            # retry once as a fresh conversation with the full transcript.
                            if not state or turn.emitted:
                                raise
                            logger.warning("Claude Code session %s expired; replaying the full transcript",
                                           state["session_id"])
                            self._checkpoint.path.unlink(missing_ok=True)
                            state = None
                        except ClaudeCodeSoftLimitNotice as exc:
                            self._raise_if_usage_limited(exc.detail or str(exc), model, cause=exc)
                            soft_limits += 1
                            if turn.emitted:
                                # Part of this answer already reached the user.
                                raise
                            if soft_limits >= _SOFT_LIMIT_ATTEMPTS:
                                raise soft_limit_exhausted(soft_limits, exc) from exc
                            logger.warning("Claude Code soft limit/rate-limit notice; retrying (attempt %d/%d): %s",
                                           soft_limits, _SOFT_LIMIT_ATTEMPTS, (exc.detail or str(exc))[:160])
                            self._discard_process()
                            self._wait(min(_SOFT_LIMIT_BASE_DELAY * soft_limits, 6.0), epoch, deadline)
                        except RuntimeError as exc:
                            # A warm process that died while idle: start one instead.
                            if turn.mode != "warm" or turn.marks or turn.emitted or type(exc) is not RuntimeError:
                                raise
                        self._discard_process()
                parked = turn.parked
                keep = None if parked else self._warm_record(turn)
                return result
            finally:
                if keep is not None:
                    self._park_warm(*keep)
                elif not parked:
                    self._cleanup()

    def _drive(self, turn, *, claims: bool):
        """Feed CLI events to ``turn`` until it returns a completion; log the turn either way.

        ``claims``: a main conversation's reply carries the pending notices (plan
        warnings, CLI model fallbacks) for the agent to show; auxiliary calls never do.
        """
        outcome = "error"
        pid = self._process.pid if self._process is not None else None
        try:
            while True:
                event = self._event(turn.epoch, turn.deadline)
                result = turn.dispatch(event)
                yield from turn.chunks
                turn.chunks.clear()
                if result is not None:
                    if claims:
                        result.usage.hermes_claim_notices = self.take_notices
                    if turn.streaming:
                        yield from final_chunks(result)
                    # Only full consumption may retain the MCP pause or publish
                    # durable state. Generator.close/error/cancel takes cleanup.
                    turn.commit()
                    outcome = "tool_call" if turn.parked else "ok"
                    return result
        except BaseException as exc:
            outcome = type(exc).__name__
            raise
        finally:
            turn.log(outcome, pid)

    def _wait(self, seconds: float, epoch: int, deadline: float) -> None:
        until = time.monotonic() + seconds
        while (left := until - time.monotonic()) > 0:
            self._check(epoch, deadline)
            time.sleep(min(0.2, left))
        self._check(epoch, deadline)

    # ── Plan limits and notices ──────────────────────────────────────

    def _record_rate_limit(self, info) -> None:
        if not isinstance(info, dict):
            return
        self._last_rate_limit = dict(info)
        for keys, text in record_rate_limit(info):
            self._add_notice(text, keys)

    def _raise_if_usage_limited(self, text: str, model: str, *, cause: BaseException) -> None:
        """Raise ``ClaudeCodeUsageLimitError`` when a limit notice cannot be retried.

        The turn's ``rate_limit_info`` decides (``rejected`` without usable extra
        usage); without it, Claude Code's own "· resets" wording does. A spend-limit
        banner alone stays a soft notice. Later calls for the model fail fast.
        """
        info = dict(self._last_rate_limit)
        _kind, resets_at = limit_window(info)
        rejected = (info.get("status") == "rejected" and not overage_usable(info)
                    and (resets_at is None or resets_at > time.time()))
        if not rejected and not HARD_LIMIT_TEXT_RE.search(text or ""):
            return
        # Stale info from an earlier turn must not name the wrong window.
        source = info if rejected else {}
        error = usage_limit_error(source, text)
        block_usage_limit(model, source, text)
        logger.warning("Claude Code subscription limit reached for %s (type=%s, resets_at=%s); not retrying, "
                       "and calls fail fast until the reset", model, error.rate_limit_type or "?",
                       error.resets_at or "?")
        raise error from cause

    def _add_notice(self, text: str, keys: tuple = ()) -> None:
        """Queue a notice; ``keys`` are the plan windows a warning is about. A notice for a
        window already pending replaces it (latest figures); an identical text merges."""
        if not text:
            return
        with self._notice_lock:
            for index, (item, item_keys) in enumerate(self._pending_notices):
                if item == text or any(key in item_keys for key in keys):
                    self._pending_notices[index] = (text, item_keys + tuple(k for k in keys if k not in item_keys))
                    return
            self._pending_notices.append((text, tuple(keys)))
            del self._pending_notices[:-_MAX_PENDING_NOTICES]

    def take_notices(self) -> list[str]:
        """Pop the notices to show with the reply just produced.

        Only a caller that shows them to the user may take them: plan warnings count
        as announced from here on. A plan warning whose window has reset meanwhile is
        dropped, and a fallback notice shown in the last ``_NOTICE_REPEAT_SECONDS`` is
        not repeated.
        """
        with self._notice_lock:
            pending, self._pending_notices = self._pending_notices, []
        now = time.time()
        pending = [(text, keys) for text, keys in pending
                   if not any(reset is not None and reset <= now for _name, reset in keys)]
        groups = [keys for _text, keys in pending if keys]
        verdicts = [True] * len(groups)
        if groups:
            try:
                verdicts = mark_limit_warnings_delivered(groups)
            except Exception:
                logger.debug("Could not record delivered Claude plan warnings", exc_info=True)
        shown, index = [], 0
        with self._notice_lock:
            self._shown_notices = {t: at for t, at in self._shown_notices.items() if now - at < _NOTICE_REPEAT_SECONDS}
            for text, keys in pending:
                if keys:
                    index += 1
                    if not verdicts[index - 1]:
                        continue
                elif text in self._shown_notices:
                    continue
                else:
                    self._shown_notices[text] = now
                shown.append(text)
        return shown

    @staticmethod
    def _compatible_model(requested: str, actual: str) -> bool:
        if requested == actual:
            return True
        return requested in ("opus", "sonnet", "haiku", "fable") and actual.startswith("claude-" + requested + "-")
