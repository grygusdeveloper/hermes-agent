"""Native stream-json event ownership and Hermes tool-result fencing.

Error semantics, plan-limit recording, quiet work turns, liveness ticks and the
per-turn telemetry line are ported from the fork's _TurnMonitor/_StreamGate and
stream parser (46d684c584, f8a124c66f, 7025953856, d6687f3f90, f0e6421c31,
9112f82f46, 232446c398, 9daf44b209, 46eb9e7345, 73d5a85c74).
"""
from __future__ import annotations

import logging
import re
import time

from agent.claude_code_checkpoint import valid_uuid
from agent.claude_code_errors import (
    ClaudeCodeAPIError, ClaudeCodeSessionExpired, ClaudeCodeSoftLimitNotice, api_error_body,
    is_expired_session_error, is_soft_limit_detail, is_soft_limit_notice, model_fallback_notice,
    refusal_error, result_error,
)
from agent.claude_code_protocol import (assistant_parts, assistant_row, completion,
                                        message_fingerprints, text_content, stream_chunk)

logger = logging.getLogger("agent.claude_code_session")

# At most one empty liveness chunk this often while Claude produces output that is
# not streamed (tool-call JSON, unsummarized thinking): the consumer's stale-stream
# detector must see a live stream.
KEEPALIVE_SECONDS = 5.0
# On tool turns prose starts streaming only past this many characters: narration
# before a tool call ("Checking the logs...") is almost always shorter, and text
# that has not started streaming when the call arrives goes out as commentary
# (the interim-message rail), never as the answer (f8a124c66f).
STREAM_COMMIT_CHARS_WITH_TOOLS = 600
REFUSED_SEPARATOR = "\n\n⚠ Claude declined to continue this reply:\n\n"
# Fences also open inside list items and block quotes; a backtick fence's info
# string holds no backtick.
_FENCE_OPEN_RE = re.compile(r"^[ \t>]*(?:(`{3,})(?![^\n]*`)|(~{3,}))")


def open_fence_closer(text: str) -> str:
    """The fence that closes the code block ``text`` ends inside, or ""."""
    fence = None
    for line in text.splitlines(keepends=True):
        if fence is None:
            opened = _FENCE_OPEN_RE.match(line)
            if opened is not None:
                marker = opened.group(1) or opened.group(2)
                fence = (marker[0], len(marker))
        elif line.endswith("\n"):
            stripped = line.lstrip(" \t>").strip()
            if len(stripped) >= fence[1] and set(stripped) == {fence[0]}:
                fence = None
    return fence[0] * fence[1] if fence is not None else ""


def _synthetic(event: dict) -> bool:
    """True when the CLI, not the model, wrote this assistant message (limit/API-error banners)."""
    message = event.get("message")
    if isinstance(message, dict) and message.get("model") == "<synthetic>":
        return True
    return bool(event.get("isApiErrorMessage") is True or event.get("is_api_error_message") is True
                or event.get("error"))


class NativeTurn:
    _HANDLERS = {
        "control_response": "on_initialized", "control_request": "on_control",
        "system": "on_system", "assistant": "on_assistant", "result": "on_result",
        "stream_event": "on_stream_event", "user": "ignore", "rate_limit_event": "on_rate_limit",
    }
    _PARTIAL_HANDLERS = {
        "content_block_start": "on_block_start", "content_block_delta": "on_block_delta",
        "content_block_stop": "on_block_stop", "message_start": "ignore",
        "message_delta": "ignore", "message_stop": "ignore", "ping": "ignore",
    }

    def __init__(self, session, *, model, messages, specs, identity, epoch, deadline, transcript, required,
                 streaming=False, mode="cold"):
        self.session = session
        self.model, self.messages, self.specs = model, messages, specs
        self.identity, self.epoch, self.deadline = identity, epoch, deadline
        self.transcript, self.required = transcript, required
        self.initialized = self.parked = False
        self.text = self.reasoning = ""
        self.calls = []
        self.tool_results = {}
        self.session_id = self.checkpoint_uuid = self.stop_reason = None
        self.historical_ids = {c["id"] for row in messages for c in row.get("tool_calls", [])}
        self.streaming = streaming
        self.chunks = []
        self.partial_text = self.partial_reasoning = ""
        self.blocks = {}
        self.publish_state = None
        # The model the turn's assistant messages name (a CLI fallback model included).
        self.served_by = None
        # Text the CLI itself wrote (``<synthetic>`` banners) and its error flag.
        self.cli_notice = ""
        self.cli_error = None
        self.refusal_no_fallback = False
        self.fallback_notice = ""
        # Quiet work turns: without tools there is nothing to hold prose for.
        self.committed = not specs
        self.held = ""
        self.shown = ""
        self.emitted = False
        self.mode = mode
        self.started = self._last_yield = time.monotonic()
        self.marks = {}
        self.api_retries = 0
        self.result_event = None
        self.published = None

    def dispatch(self, event):
        handler = self._HANDLERS.get(event["type"])
        if handler is None:
            raise RuntimeError("Unsupported Claude native protocol event")
        before = len(self.chunks)
        result = getattr(self, handler)(event)
        if self.streaming and self.initialized and result is None and len(self.chunks) == before:
            now = time.monotonic()
            if now - self._last_yield >= KEEPALIVE_SECONDS:
                self._last_yield = now
                self.chunks.append(stream_chunk(self.model))
        return result

    def ignore(self, event):
        # A user/tool_result echo is historical data, never a new executable
        # call. Only successful result + exhausted consumption publishes state.
        return None

    def _mark(self, key):
        self.marks.setdefault(key, time.monotonic())

    def _chunk(self, **fields):
        self.chunks.append(stream_chunk(self.model, **fields))
        self.emitted = True
        self._last_yield = time.monotonic()
        self.shown += fields.get("text") or ""

    def _stream_text(self, value):
        if self.committed:
            self._chunk(text=value)
            return
        self.held += value
        if len(self.held) >= STREAM_COMMIT_CHARS_WITH_TOOLS:
            self.committed = True
            value, self.held = self.held, ""
            self._chunk(text=value)

    def _hold_as_commentary(self):
        """Narration before tool calls: message content, never the answer stream."""
        if self.held:
            value, self.held = self.held, ""
            self._chunk(commentary=value)

    def _release_held(self):
        if self.held:
            value, self.held = self.held, ""
            self._chunk(text=value)

    def _delta(self, kind, value):
        value = text_content(value)
        if kind == "text":
            self.partial_text += value
        else:
            self.partial_reasoning += value
        if value:
            self._mark(kind)
        if self.streaming and value:
            if kind == "text":
                self._stream_text(value)
            else:
                self._chunk(reasoning=value)

    def on_stream_event(self, event):
        if not self.initialized or event.get("parent_tool_use_id"):
            raise RuntimeError("Claude CLI emitted an unowned partial event")
        if event.get("session_id") and event["session_id"] != self.session_id:
            raise RuntimeError("Claude CLI changed partial session identity")
        self._mark("event")
        partial = event.get("event", {})
        handler = self._PARTIAL_HANDLERS.get(partial.get("type"))
        if handler is None:
            raise RuntimeError("Unsupported Claude partial protocol event")
        return getattr(self, handler)(partial)

    def on_block_start(self, partial):
        index, block = partial.get("index"), partial.get("content_block", {})
        if not isinstance(index, int) or index in self.blocks:
            raise RuntimeError("Claude CLI returned an invalid partial block index")
        kind = block.get("type")
        if kind not in ("text", "thinking", "tool_use", "redacted_thinking"):
            raise RuntimeError("Unsupported Claude partial content block")
        self.blocks[index] = kind
        if kind in ("thinking", "redacted_thinking"):
            self._mark("thinking")
        if kind == "tool_use" and self.streaming:
            self._hold_as_commentary()
        if kind in ("text", "thinking"):
            self._delta(kind, block.get("text" if kind == "text" else "thinking", ""))

    def on_block_delta(self, partial):
        delta = partial.get("delta", {})
        pair = (self.blocks.get(partial.get("index")), delta.get("type"))
        if pair == ("text", "text_delta"):
            self._delta("text", delta.get("text"))
        elif pair == ("thinking", "thinking_delta"):
            self._delta("thinking", delta.get("thinking"))
        elif pair not in (("tool_use", "input_json_delta"), ("thinking", "signature_delta")):
            raise RuntimeError("Claude CLI returned an invalid partial block delta")
        # Partial tool JSON never enters self.calls or the generic tool accumulator.

    def on_block_stop(self, partial):
        if partial.get("index") not in self.blocks:
            raise RuntimeError("Claude CLI stopped an unknown partial block")
        self.blocks.pop(partial["index"])

    def on_initialized(self, event):
        response = event.get("response", {})
        if self.initialized or response.get("request_id") != "hermes-init" or response.get("subtype") != "success":
            raise RuntimeError("Claude CLI rejected the native initialization protocol")
        self.initialized = True
        self.session._send({"type": "user", "message": {"role": "user", "content": self.transcript}})

    def on_rate_limit(self, event):
        self.session._record_rate_limit(event.get("rate_limit_info"))

    def supply_result(self, event, call):
        row = self.tool_results.pop(call["id"])
        self.session._mcp_reply(event, {"content": [{"type": "text", "text": text_content(row.get("content"))}],
                                        "isError": bool(row.get("is_error", False))})
        self.calls.remove(call)

    def on_control(self, event):
        pending = self.session._control(event, self.specs, self.calls)
        if pending is None:
            return None
        if self.tool_results:
            self.supply_result(event, pending["call"])
            return None
        if self.streaming:
            self._hold_as_commentary()
        row = assistant_row(self.text, self.calls)
        self.session._pending = {**pending, "calls": self.calls, "identity": self.identity, "epoch": self.epoch,
                                 "session_id": self.session_id,
                                 "blocks": dict(self.blocks),
                                 "fingerprints": message_fingerprints(self.messages + [row])}
        self.parked = True
        return completion(self.model, self.text, self.reasoning, self.calls)

    def on_system(self, event):
        subtype = event.get("subtype")
        if subtype == "init":
            self._mark("init")
            self.session_id = valid_uuid(event.get("session_id"))
            if event.get("model") and not self.session._compatible_model(self.model, event["model"]):
                raise RuntimeError("Claude CLI silently selected a different model")
        elif subtype in ("model_fallback", "model_refusal_fallback", "model_refusal_no_fallback"):
            # CLI-side fallbacks are notices, not failures; a refusal without a
            # fallback model ends the turn through its result (46eb9e7345, d6687f3f90).
            logger.warning("Claude Code %s: %s -> %s (scope=%s, category=%s)", subtype,
                           event.get("originalModel") or event.get("original_model"),
                           event.get("fallbackModel") or event.get("fallback_model") or "-",
                           event.get("scope") or "session",
                           event.get("apiRefusalCategory") or event.get("api_refusal_category"))
            if subtype == "model_refusal_no_fallback":
                self.refusal_no_fallback = True
            else:
                self.fallback_notice = model_fallback_notice(subtype, event)
        elif subtype == "api_retry":
            self.api_retries += 1
            logger.warning("Claude Code API retry %s/%s (status=%s) in %sms", event.get("attempt"),
                           event.get("max_retries"), event.get("error_status"), event.get("retry_delay_ms"))
        elif subtype == "compact_boundary":
            logger.warning("Claude Code compacted its conversation (session=%s)", self.session_id)
        elif subtype == "error":
            raise RuntimeError("Claude CLI reported a native initialization error")

    def on_assistant(self, event):
        if not self.initialized:
            raise RuntimeError("Claude CLI emitted assistant content before native initialization")
        if _synthetic(event):
            # A limit or API-error banner the CLI wrote: never model content.
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            text = "".join(str(b.get("text") or "") for b in message.get("content") or []
                           if isinstance(b, dict) and b.get("type") == "text").strip()
            self.cli_notice = text or self.cli_notice
            self.cli_error = event.get("error") or message.get("error") or self.cli_error
            return
        new_text, new_reasoning, new_calls = assistant_parts(event, self.specs,
            self.historical_ids | {c["id"] for c in self.calls})
        if not new_text.startswith(self.partial_text) or not new_reasoning.startswith(self.partial_reasoning):
            raise RuntimeError("Claude assistant disagrees with streamed content")
        if self.streaming:
            text_tail = new_text[len(self.partial_text):]
            reasoning_tail = new_reasoning[len(self.partial_reasoning):]
            if reasoning_tail:
                self._chunk(reasoning=reasoning_tail)
            if text_tail:
                self._stream_text(text_tail)
            if new_calls:
                self._hold_as_commentary()
        self.partial_text = self.partial_reasoning = ""
        # CLI 2.1.277 delivers the authoritative assistant BEFORE the last
        # content_block_stop. Keep lifecycle bookkeeping until that event.
        self.text += new_text
        self.reasoning += new_reasoning
        self.calls += new_calls
        self.checkpoint_uuid = valid_uuid(event.get("uuid"))
        self.stop_reason = event.get("message", {}).get("stop_reason")
        self.served_by = event.get("message", {}).get("model") or self.served_by

    def _failure(self, event):
        """The typed error for an unsuccessful result, from what the CLI itself wrote."""
        status, code, text = result_error(event)
        errors = event.get("errors") if isinstance(event.get("errors"), list) else []
        detail = (text or "; ".join(str(e) for e in errors)[:300] or self.cli_notice[:300]
                  or f"subtype={event.get('subtype')!r}")
        if event.get("stop_reason") == "refusal" or self.refusal_no_fallback:
            return refusal_error(event.get("result") or self.cli_notice)
        if is_expired_session_error(detail) or any(is_expired_session_error(str(e)) for e in errors):
            return ClaudeCodeSessionExpired(f"Claude Code result error: {detail}")
        body = api_error_body(status, code, detail) if status or code else None
        if (status == 429 or self.cli_error == "rate_limit" or is_soft_limit_detail(detail)
                or is_soft_limit_notice(self.cli_notice)):
            return ClaudeCodeSoftLimitNotice(f"Claude Code rate-limit result: {detail}", status_code=status,
                                             body=body, detail=self.cli_notice or detail)
        return ClaudeCodeAPIError(f"Claude Code result rejected (subtype={event.get('subtype')!r}): {detail}",
                                  status_code=status, body=body, detail=detail)

    def on_result(self, event):
        if not self.initialized:
            raise RuntimeError("Claude CLI returned an unsuccessful native result")
        self.result_event = event
        if event.get("is_error") or event.get("subtype") != "success":
            error = self._failure(event)
            shown = self.shown
            if getattr(error, "body", {}).get("error", {}).get("type") == "refusal" and self.streaming and shown.strip():
                # Part of the refused reply is on screen: end the stream as a
                # refusal (content_filter) with the CLI's explanation, never an
                # error Hermes would continue as a cut-off reply. Nothing publishes.
                closer = open_fence_closer(shown)
                self.held = ""
                self._chunk(text=(f"\n{closer}" if closer else "") + REFUSED_SEPARATOR + error.detail)
                logger.warning("Claude Code refused after %d streamed chars; ending the stream as a refusal",
                               len(shown))
                self.text = self.shown
                return completion(self.model, self.text, self.reasoning, [], event.get("usage"), "refusal")
            raise error
        if self.calls or self.tool_results:
            raise RuntimeError("Claude CLI completed tool_use without a correlated Hermes tool call")
        if self.blocks or self.partial_text or self.partial_reasoning:
            raise RuntimeError("Claude CLI completed without an authoritative assistant message")
        if self.required:
            raise RuntimeError("Claude CLI did not honor required tool choice")
        if not self.text and not self.reasoning:
            if self.cli_notice:
                # Only a CLI banner instead of an answer (9112f82f46).
                status, code, _text = result_error(event)
                body = api_error_body(status, code, self.cli_notice[:300]) if status or code else None
                kind = ClaudeCodeSoftLimitNotice if is_soft_limit_notice(self.cli_notice) else ClaudeCodeAPIError
                raise kind(f"Claude Code returned a notice instead of an answer: {self.cli_notice[:300]}",
                           status_code=status, body=body, detail=self.cli_notice)
            raise RuntimeError("Claude CLI returned no assistant content")
        sid = valid_uuid(event.get("session_id"))
        if self.session_id and sid != self.session_id:
            raise RuntimeError("Claude CLI changed session identity during a request")
        if self.streaming:
            self._release_held()
        result = completion(self.model, self.text, self.reasoning, [], event.get("usage"), self.stop_reason,
                            event.get("modelUsage"), self.served_by)
        self.publish_state = (sid, valid_uuid(self.checkpoint_uuid))
        return result

    def commit(self):
        with self.session._guard:
            self.session._check(self.epoch, self.deadline)
            if self.publish_state and self.session._checkpoint:
                sid, checkpoint = self.publish_state
                fingerprints = message_fingerprints(self.messages + [assistant_row(self.text, [])])
                self.session._checkpoint.publish(self.identity, sid, checkpoint, fingerprints)
                # What a process kept warm for the next user turn holds.
                self.published = (self.identity, sid, checkpoint, fingerprints)
        if self.fallback_notice:
            self.session._add_notice(self.fallback_notice)

    def log(self, outcome, pid=None):
        """One INFO line per CLI turn: cold/resumed/tool continuation, latencies, tokens."""
        def since(key):
            mark = self.marks.get(key)
            return str(int((mark - self.started) * 1000)) if mark is not None else "-"

        result = self.result_event or {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        logger.info(
            "Claude Code turn: mode=%s pid=%s outcome=%s total_ms=%d init_ms=%s first_event_ms=%s "
            "thinking_ms=%s text_ms=%s ttft_ms=%s api_ms=%s out_tokens=%s cache_read=%s cache_write=%s "
            "payload_chars=%d api_retries=%d",
            self.mode, pid if pid is not None else "-", outcome, int((time.monotonic() - self.started) * 1000),
            since("init"), since("event"), since("thinking"), since("text"),
            result.get("ttft_ms", "-"), result.get("duration_api_ms", "-"), usage.get("output_tokens", "-"),
            usage.get("cache_read_input_tokens", "-"), usage.get("cache_creation_input_tokens", "-"),
            len(self.transcript) if isinstance(self.transcript, str) else sum(
                len(b.get("text") or b.get("source", {}).get("data", "")) for b in self.transcript),
            self.api_retries,
        )
