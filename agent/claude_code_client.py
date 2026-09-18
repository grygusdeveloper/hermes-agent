"""OpenAI-compatible shim that forwards Hermes requests to the Claude Code CLI.

This adapter lets Hermes treat the locally-installed, already-authenticated
``claude`` (Claude Code CLI 2.x) as a chat-style backend — analogous to
``agent/copilot_acp_client.py`` (the GitHub Copilot / Antigravity ACP bridge)
but using Claude Code's **native stream-json stdin transport** instead of an
argv-bound prompt.

Key differences from the Copilot ACP bridge
-------------------------------------------
* **Real system slot** — the Hermes system prompt, tool protocol and tool
  schemas travel via ``--system-prompt-file``; the transcript (with native
  base64 image blocks) travels as a stream-json user message on stdin, never
  as an ``execve`` argument.  There is no ``MAX_ARG_STRLEN`` ceiling.
* **Durable session continuation** — one Claude Code ``session_id`` per Hermes
  agent (``agent.portal_tags.get_bridge_state_key``); later turns send only
  the incremental new messages, to a warm process kept alive between the
  model calls of a tool loop, or via ``--resume`` at the last published
  checkpoint in a new process.
* **Live streaming** — ``--include-partial-messages`` thinking deltas stream
  as reasoning; prose streams once it cannot be a retried banner/preamble.
* **Distinct identity** — provider ``claude-code``, base marker
  ``acp://claude-code``.  Does not reuse, relabel, or replace ``copilot-acp``
  or the Antigravity command.
* **Native tools disabled** — ``--tools ""`` turns off all Claude Code built-in
  tools so every tool call remains under Hermes logging, permissions, MCP, and
  approvals.  Hermes injects its tool schemas into the prompt and parses
  ``<tool_call>`` blocks back out of the response (same proven parser as the
  Copilot ACP bridge).

Security
--------
Authentication is delegated entirely to the already-authenticated Claude Code
CLI.  This module never reads, passes, or logs credentials.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.claude_code_session import (
    _HERMES_BACKEND_SYSTEM_PROMPT,
    ClaudeCodeSession,
    _render_content_collecting_images,
    _ToolNameResolver,
)
from agent.portal_tags import get_bridge_state_key

CLAUDE_CODE_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_PROMPT_FORMAT_VERSION = 5

# Tool-call extraction shared with the Copilot ACP bridge (same <tool_call> shape).
from agent.copilot_acp_client import (  # noqa: E402
    _extract_tool_calls_from_text,
    _render_message_content,
)


def _render_transcript_content(content: Any, images: list[dict[str, Any]]) -> str:
    if isinstance(content, list):
        return _render_content_collecting_images(content, images).strip()
    return _render_message_content(content)


def _build_claude_code_request(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Split a Hermes request into ``(system_prompt, transcript, images)``.

    The system prompt carries the backend contract, the tool protocol + tool
    schemas and Hermes's own leading system messages, so Claude receives them
    in the real system slot (``--system-prompt-file``). The transcript keeps
    assistant ``tool_calls`` and tool-result ``name``/``tool_call_id`` linkage
    so fresh/expired full replays remain semantically complete. Image parts
    become native image blocks, referenced from the text as ``[Image #n]``.
    """

    import json

    sections: list[str] = [
        _HERMES_BACKEND_SYSTEM_PROMPT,
        "You are the active reasoning model inside Hermes.",
        "Hermes, not Claude Code, owns all tool execution. Claude Code native tools are "
        "intentionally disabled; every tool listed below remains available through Hermes.",
        "TOOL RULE: If you take an action, emit one or more "
        "<tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape. "
        "For multi-step work, you may precede the first tool batch with one short, concrete "
        "user-facing progress sentence saying what you are checking and naming any skill "
        "you load. This is activity reporting, not private chain-of-thought. Do not narrate "
        "every trivial call or append a final answer to a tool-call turn. Independent calls "
        "may be emitted together to reduce round trips.",
        "FINALITY RULE: If no tool is needed, return the complete user-facing answer now. "
        "Never end with process narration such as 'I will check' or 'let me inspect'. "
        "Do not repeat an inspection whose result is already present in the transcript.",
        "STYLE RULE: In final answers, sound like a natural, thoughtful collaborator. "
        "Match the user's language, tone, and requested level of detail. Avoid walls of text. "
        "When an answer has several findings, decisions, comparisons, or steps, format it as "
        "a readable Markdown document with short descriptive headings, compact paragraphs, "
        "and bullets or numbered steps where useful. Use bold labels sparingly. Icons are "
        "optional and should add a real visual cue, never decorate every line. Keep simple "
        "conversation as natural prose rather than forcing a rigid report template. Lead "
        "with the useful conclusion, not process commentary.",
        "IMAGE RULE: Images attached to a message are sent as native image inputs labelled "
        "'Image #n:' and referenced in the transcript as [Image #n]. Look at them directly.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "For each tool call, emit <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    images: list[dict[str, Any]] = []
    leading_system: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        if str(message.get("role") or "").strip().lower() != "system":
            break
        rendered = _render_message_content(message.get("content"))
        if rendered:
            leading_system.append(rendered)
        index += 1
    if leading_system:
        sections.append(
            "Hermes system instructions (authoritative policy for this conversation):\n\n"
            + "\n\n".join(leading_system)
        )

    transcript: list[str] = []
    # Tool results reloaded from the session DB carry no ``name``; recover it
    # from the preceding assistant call so replays label results like live
    # turns.
    tool_names = _ToolNameResolver(messages[:index])
    for message in messages[index:]:
        if not isinstance(message, dict):
            continue
        tool_name = tool_names.feed(message)
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("id") or ""
            rendered = _render_transcript_content(message.get("content"), images)
            meta = []
            if isinstance(tool_name, str) and tool_name.strip():
                meta.append(f"name={tool_name.strip()}")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                meta.append(f"tool_call_id={tool_call_id.strip()}")
            header = "Tool Result"
            if meta:
                header = f"Tool Result ({', '.join(meta)})"
            transcript.append(f"{header}:\n{rendered}")
            continue
        if role not in {"system", "user", "assistant"}:
            role = "context"

        rendered = _render_transcript_content(message.get("content"), images)
        tool_calls = message.get("tool_calls")
        call_blocks: list[str] = []
        if tool_calls:
            for tc in tool_calls if isinstance(tool_calls, list) else []:
                try:
                    if hasattr(tc, "model_dump"):
                        obj = tc.model_dump()
                    elif isinstance(tc, dict):
                        obj = tc
                    else:
                        obj = {
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", None),
                                "arguments": getattr(
                                    getattr(tc, "function", None), "arguments", "{}"
                                ),
                            },
                        }
                    call_blocks.append(
                        "<tool_call>"
                        + json.dumps(obj, ensure_ascii=False)
                        + "</tool_call>"
                    )
                except Exception:
                    continue
        body_parts = [p for p in (rendered, "\n".join(call_blocks)) if p]
        if not body_parts:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n" + "\n".join(body_parts))

    prompt_sections: list[str] = []
    if transcript:
        prompt_sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    prompt_sections.append("Continue the conversation from the latest user request.")

    system_prompt = "\n\n".join(s.strip() for s in sections if s and s.strip())
    prompt_text = "\n\n".join(s.strip() for s in prompt_sections if s and s.strip())
    return system_prompt, prompt_text, images


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Flat text view of :func:`_build_claude_code_request` (system + transcript)."""

    system_prompt, prompt_text, _images = _build_claude_code_request(
        messages, model=model, tools=tools, tool_choice=tool_choice
    )
    return system_prompt + "\n\n" + prompt_text


class _ClaudeCodeChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCodeChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _ClaudeCodeChatCompletions(client)


class ClaudeCodeClient:
    """Minimal OpenAI-client-compatible facade for the Claude Code CLI.

    Mirrors the shape of ``CopilotACPClient`` so the rest of Hermes
    (conversation loop, streaming, auxiliary client) can treat it uniformly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code"
        self.base_url = base_url or CLAUDE_CODE_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        # ``command``/``args`` are accepted for interface parity with
        # CopilotACPClient; only the command path is honoured (Claude Code's
        # transport is fixed: stream-json in/out, tools disabled).
        self._claude_command = command or _resolve_command()
        self._claude_cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _ClaudeCodeChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        # One durable Claude Code session per cached Hermes model client, so
        # keeping the conversation here prevents cross-thread collisions even
        # when two spawned workspaces begin with identical prompts.
        self._claude_session = ClaudeCodeSession()
        self._owns_claude_session = True

    def close(self) -> None:
        if self._owns_claude_session:
            self._claude_session.shutdown()
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        system_prompt, prompt_text, prompt_images = _build_claude_code_request(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        # Prefer per-request effort kwargs, then configured agent effort.
        effort = _resolve_effort_from_kwargs(kwargs) or _resolve_effort()
        tools_digest = _tools_digest(tools, tool_choice=tool_choice)

        # Durable Claude Code continuity is scoped per Hermes agent (main
        # agent, each delegate child); background-review forks publish None.
        # Auxiliary title/compression/vision calls have no tool schema: they
        # stay stateless and never touch an agent's conversation mapping.
        state_key = get_bridge_state_key() if tools else None

        run_kwargs: dict[str, Any] = dict(
            messages=messages or [],
            model=model or "sonnet",
            effort=effort,
            tools_digest=tools_digest,
            timeout_seconds=_effective_timeout,
            cwd=self._claude_cwd,
            env=_build_subprocess_env(),
            state_key=state_key,
            command=self._claude_command,
            system_prompt=system_prompt,
            # A tool loop follows only when tools were offered; keep the
            # process warm for it whether or not the state is durable.
            keepalive=bool(tools),
        )
        if prompt_images:
            run_kwargs["prompt_images"] = prompt_images

        if stream:
            return self._stream_chat_completion(
                prompt_text=prompt_text,
                run_kwargs=run_kwargs,
                model=model or "sonnet",
            )

        response_text, reasoning_text = self._claude_session.run(prompt_text, **run_kwargs)

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = _completion_usage(self._claude_session.last_usage)
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-code",
        )
        return completion

    def _stream_chat_completion(
        self,
        *,
        prompt_text: str,
        run_kwargs: dict[str, Any],
        model: str,
    ):
        """Yield OpenAI-style stream chunks as Claude Code output arrives.

        Thinking streams live as ``reasoning_content``. Prose streams live once
        the session's gate has validated the attempt (never raw ``<tool_call>``
        markup). Contract: concatenated ``delta.content`` over the whole stream
        equals the non-streaming ``cleaned_text``.
        """

        import queue

        q: queue.Queue[Any] = queue.Queue()
        done = object()
        error_box: dict[str, BaseException] = {}

        def on_text(text: str) -> None:
            if text:
                q.put(("text", text))

        def on_reasoning(text: str) -> None:
            if text:
                q.put(("reasoning", text))

        def worker() -> None:
            try:
                response_text, reasoning_text = self._claude_session.run(
                    prompt_text,
                    on_text_chunk=on_text,
                    on_reasoning_chunk=on_reasoning,
                    **run_kwargs,
                )
                q.put(("final", response_text, reasoning_text))
            except BaseException as exc:  # noqa: BLE001 — surface to consumer
                error_box["exc"] = exc
            finally:
                q.put(done)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def _chunk(*, content=None, reasoning=None, tool_calls=None, finish_reason=None, role="assistant"):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            role=role,
                            content=content,
                            tool_calls=tool_calls,
                            reasoning_content=reasoning,
                            reasoning=reasoning,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                model=model,
                usage=None,
            )

        emitted_text: list[str] = []
        streamed_reasoning = False
        final_text = ""
        final_reasoning = ""
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                if not (isinstance(item, tuple) and item):
                    continue
                if item[0] == "text":
                    emitted_text.append(item[1])
                    yield _chunk(content=item[1])
                elif item[0] == "reasoning":
                    streamed_reasoning = True
                    yield _chunk(reasoning=item[1])
                elif item[0] == "final":
                    final_text = item[1] or ""
                    final_reasoning = item[2] or ""
            if error_box.get("exc"):
                raise error_box["exc"]

            tool_calls, cleaned = _extract_tool_calls_from_text(final_text or "")
            already = "".join(emitted_text)
            # Safety net: emit any cleaned remainder the session did not stream.
            if cleaned and cleaned != already and cleaned.startswith(already):
                yield _chunk(content=cleaned[len(already):])
            # Reasoning reaches the consumer exactly once: live, or here.
            late_reasoning = None if streamed_reasoning else (final_reasoning or None)

            finish = "tool_calls" if tool_calls else "stop"
            if tool_calls:
                deltas = []
                for index, tc in enumerate(tool_calls):
                    deltas.append(
                        SimpleNamespace(
                            index=index,
                            id=getattr(tc, "id", None),
                            type=getattr(tc, "type", "function"),
                            function=SimpleNamespace(
                                name=getattr(tc.function, "name", None),
                                arguments=getattr(tc.function, "arguments", None),
                            ),
                        )
                    )
                # Never re-emit prose on the finish chunk.
                yield _chunk(tool_calls=deltas, reasoning=late_reasoning, finish_reason=finish)
            else:
                if late_reasoning:
                    yield _chunk(reasoning=late_reasoning)
                yield _chunk(role=None, finish_reason=finish)
            yield SimpleNamespace(
                choices=[],
                model=model,
                usage=_completion_usage(self._claude_session.last_usage),
            )
        finally:
            thread.join(timeout=5)



def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_PATH", "").strip()
        or os.getenv("CLAUDE_BIN", "").strip()
        or "claude"
    )


def _tools_digest(
    tools: list[dict[str, Any]] | None,
    *,
    tool_choice: Any = None,
) -> str:
    """Stable identity for the complete Hermes tool surface.

    Hashes canonical tool schemas (not just names), tool_choice, and a
    prompt-format version so schema/description changes force a fresh
    Claude Code session instead of resuming with stale tool instructions.
    """

    import hashlib
    import json as _json

    def _normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(k): _normalize(v)
                for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_normalize(item) for item in value]
        return str(value)

    specs: list[Any] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        specs.append(_normalize(tool))
    # Tool registry order can vary across gateway turns even when the actual
    # capability set is identical. Order is not semantically meaningful, so
    # canonicalize it to avoid throwing away a warm Claude session.
    specs.sort(
        key=lambda item: _json.dumps(
            item, separators=(",", ":"), ensure_ascii=True, sort_keys=True
        )
    )
    payload = {
        "format_version": _PROMPT_FORMAT_VERSION,
        "tools": specs,
        "tool_choice": _normalize(tool_choice),
    }
    canonical = _json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _completion_usage(raw: dict[str, Any] | None) -> Any:
    """Expose Claude Code result usage through the OpenAI-compatible facade."""

    usage = raw or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    cache_write_tokens = int(usage.get("cache_write_tokens") or 0)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        cache_read_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_input_tokens=cached_tokens,
        cache_creation_input_tokens=cache_write_tokens,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or completion_tokens),
        total_cost_usd=usage.get("total_cost_usd"),
        service_tier=usage.get("service_tier"),
    )


def _resolve_effort_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    """Pull effort from OpenAI-style / Hermes kwargs when present."""

    for key in ("effort", "reasoning_effort"):
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"low", "medium", "high", "xhigh", "max", "ultracode"}:
                return normalized
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        value = reasoning.get("effort")
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"low", "medium", "high", "xhigh", "max", "ultracode"}:
                return normalized
    return None


def _resolve_effort() -> str | None:
    """Resolve the configured reasoning effort for the active agent, if any.

    Reads the agent's configured ``reasoning_effort`` without importing the
    full agent runtime (keeps this module import-light).
    """

    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        agent_cfg = config.get("agent") or {}
        if isinstance(agent_cfg, dict):
            effort = agent_cfg.get("reasoning_effort")
            if isinstance(effort, str) and effort.strip():
                normalized = effort.strip().lower()
                # Claude Code also exposes Ultracode as its workflow-aware top tier.
                if normalized in {"low", "medium", "high", "xhigh", "max", "ultracode"}:
                    return normalized
    except Exception:
        pass
    return None


def _build_subprocess_env() -> dict[str, str]:
    """Build the child-process environment.

    Authentication is delegated to the already-authenticated Claude Code CLI
    (OAuth/keychain under the CLI's own home).  Do **not** inherit Hermes
    provider API keys — that would leak credentials into the child and can
    make Claude Code prefer ANTHROPIC_API_key billing over the user's
    subscription OAuth.
    """

    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    home = os.environ.get("HOME", "").strip()
    if home:
        env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    # Defense in depth: never pass Anthropic/OpenAI provider secrets even if
    # a future scrubber change re-admits them.
    for key in list(env):
        upper = key.upper()
        if upper.startswith("ANTHROPIC_") or upper in {
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_API_KEY",
        }:
            env.pop(key, None)
    return env
