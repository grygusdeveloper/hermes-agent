"""Native stream-json event ownership and Hermes tool-result fencing."""
from __future__ import annotations

from agent.claude_code_checkpoint import valid_uuid
from agent.claude_code_protocol import (assistant_parts, assistant_row, completion,
                                        message_fingerprints, text_content, stream_chunk)


class NativeTurn:
    _HANDLERS = {
        "control_response": "on_initialized", "control_request": "on_control",
        "system": "on_system", "assistant": "on_assistant", "result": "on_result",
        "stream_event": "on_stream_event", "user": "ignore", "rate_limit_event": "ignore",
    }
    _PARTIAL_HANDLERS = {
        "content_block_start": "on_block_start", "content_block_delta": "on_block_delta",
        "content_block_stop": "on_block_stop", "message_start": "ignore",
        "message_delta": "ignore", "message_stop": "ignore", "ping": "ignore",
    }

    def __init__(self, session, *, model, messages, specs, identity, epoch, deadline, transcript, required,
                 streaming=False):
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

    def dispatch(self, event):
        handler = self._HANDLERS.get(event["type"])
        if handler is None:
            raise RuntimeError("Unsupported Claude native protocol event")
        return getattr(self, handler)(event)

    def ignore(self, event):
        # A user/tool_result echo is historical data, never a new executable
        # call. Only successful result + exhausted consumption publishes state.
        return None

    def _delta(self, kind, value):
        value = text_content(value)
        if kind == "text":
            self.partial_text += value
        else:
            self.partial_reasoning += value
        if self.streaming and value:
            self.chunks.append(stream_chunk(self.model, text=value if kind == "text" else None,
                                             reasoning=value if kind == "thinking" else None))

    def on_stream_event(self, event):
        if not self.initialized or event.get("parent_tool_use_id"):
            raise RuntimeError("Claude CLI emitted an unowned partial event")
        if event.get("session_id") and event["session_id"] != self.session_id:
            raise RuntimeError("Claude CLI changed partial session identity")
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
            self.session_id = valid_uuid(event.get("session_id"))
            if event.get("model") and not self.session._compatible_model(self.model, event["model"]):
                raise RuntimeError("Claude CLI silently selected a different model")
        elif subtype in ("model_fallback", "error"):
            raise RuntimeError("Claude CLI reported a native model or initialization error")

    def on_assistant(self, event):
        if not self.initialized:
            raise RuntimeError("Claude CLI emitted assistant content before native initialization")
        new_text, new_reasoning, new_calls = assistant_parts(event, self.specs,
            self.historical_ids | {c["id"] for c in self.calls})
        if not new_text.startswith(self.partial_text) or not new_reasoning.startswith(self.partial_reasoning):
            raise RuntimeError("Claude assistant disagrees with streamed content")
        if self.streaming:
            text_tail = new_text[len(self.partial_text):]
            reasoning_tail = new_reasoning[len(self.partial_reasoning):]
            if text_tail or reasoning_tail:
                self.chunks.append(stream_chunk(self.model, text=text_tail or None, reasoning=reasoning_tail or None))
        self.partial_text = self.partial_reasoning = ""
        # CLI 2.1.277 delivers the authoritative assistant BEFORE the last
        # content_block_stop. Keep lifecycle bookkeeping until that event.
        self.text += new_text
        self.reasoning += new_reasoning
        self.calls += new_calls
        self.checkpoint_uuid = valid_uuid(event.get("uuid"))
        self.stop_reason = event.get("message", {}).get("stop_reason")
        self.served_by = event.get("message", {}).get("model") or self.served_by

    def on_result(self, event):
        if not self.initialized or event.get("is_error") or event.get("subtype") != "success":
            raise RuntimeError("Claude CLI returned an unsuccessful native result")
        if self.calls or self.tool_results:
            raise RuntimeError("Claude CLI completed tool_use without a correlated Hermes tool call")
        if self.blocks or self.partial_text or self.partial_reasoning:
            raise RuntimeError("Claude CLI completed without an authoritative assistant message")
        if self.required:
            raise RuntimeError("Claude CLI did not honor required tool choice")
        if not self.text and not self.reasoning:
            raise RuntimeError("Claude CLI returned no assistant content")
        sid = valid_uuid(event.get("session_id"))
        if self.session_id and sid != self.session_id:
            raise RuntimeError("Claude CLI changed session identity during a request")
        result = completion(self.model, self.text, self.reasoning, [], event.get("usage"), self.stop_reason,
                            event.get("modelUsage"), self.served_by)
        self.publish_state = (sid, valid_uuid(self.checkpoint_uuid))
        return result

    def commit(self):
        with self.session._guard:
            self.session._check(self.epoch, self.deadline)
            if self.publish_state and self.session._checkpoint:
                sid, checkpoint = self.publish_state
                self.session._checkpoint.publish(self.identity, sid, checkpoint,
                    message_fingerprints(self.messages + [assistant_row(self.text, [])]))
