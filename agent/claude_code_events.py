"""Native stream-json event ownership and Hermes tool-result fencing."""
from __future__ import annotations

from agent.claude_code_checkpoint import valid_uuid
from agent.claude_code_protocol import (assistant_parts, assistant_row, completion,
                                        message_fingerprints, text_content)


class NativeTurn:
    _HANDLERS = {
        "control_response": "on_initialized", "control_request": "on_control",
        "system": "on_system", "assistant": "on_assistant", "result": "on_result",
        "stream_event": "ignore", "user": "ignore", "rate_limit_event": "ignore",
    }

    def __init__(self, session, *, model, messages, specs, identity, epoch, deadline, transcript, required):
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

    def dispatch(self, event):
        handler = self._HANDLERS.get(event["type"])
        if handler is None:
            raise RuntimeError("Unsupported Claude native protocol event")
        return getattr(self, handler)(event)

    def ignore(self, event):
        # A user/tool_result echo is historical data, never a new executable
        # call. Partial events cannot publish output before the result fence.
        return None

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
        self.text += new_text
        self.reasoning += new_reasoning
        self.calls += new_calls
        self.checkpoint_uuid = valid_uuid(event.get("uuid"))
        self.stop_reason = event.get("message", {}).get("stop_reason")

    def on_result(self, event):
        if not self.initialized or event.get("is_error") or event.get("subtype") != "success":
            raise RuntimeError("Claude CLI returned an unsuccessful native result")
        if self.calls or self.tool_results:
            raise RuntimeError("Claude CLI completed tool_use without a correlated Hermes tool call")
        if self.required:
            raise RuntimeError("Claude CLI did not honor required tool choice")
        if not self.text and not self.reasoning:
            raise RuntimeError("Claude CLI returned no assistant content")
        sid = valid_uuid(event.get("session_id"))
        if self.session_id and sid != self.session_id:
            raise RuntimeError("Claude CLI changed session identity during a request")
        result = completion(self.model, self.text, self.reasoning, [], event.get("usage"), self.stop_reason)
        with self.session._guard:
            self.session._check(self.epoch, self.deadline)
            if self.session._checkpoint:
                self.session._checkpoint.publish(self.identity, sid, valid_uuid(self.checkpoint_uuid),
                    message_fingerprints(self.messages + [assistant_row(self.text, [])]))
        return result
