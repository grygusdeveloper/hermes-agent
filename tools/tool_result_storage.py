"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import hashlib
import logging
import os
import re
import shlex
import uuid

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120
OOB_USER_MESSAGE_KEY = "_hermes_oob_user_message"
PERSISTED_OUTPUT_METADATA_KEY = "_hermes_persisted_output"
POST_PERSIST_SUFFIXES_KEY = "_hermes_post_persist_suffixes"


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def _trusted_persistence_metadata(message: dict) -> dict | None:
    """Return framework-authenticated persistence metadata, if complete."""

    metadata = message.get(PERSISTED_OUTPUT_METADATA_KEY)
    if not isinstance(metadata, dict):
        return None
    path = metadata.get("path")
    original_size = metadata.get("original_size")
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    if not isinstance(original_size, int) or original_size < 0:
        return None
    return {"path": path, "original_size": original_size}


def _compact_persisted_message(content: str, metadata: dict) -> str:
    """Drop a persisted preview using only authenticated durable metadata."""

    original_size = metadata["original_size"]
    size_kb = original_size / 1024
    size_str = (
        f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    )
    lines = [PERSISTED_OUTPUT_TAG]
    lines.append(
        f"This tool result was too large ({original_size:,} characters, {size_str})."
    )
    lines.extend(
        (
            f"Full output saved to: {metadata['path']}",
            "Use read_file with offset and limit to inspect the full output.",
            PERSISTED_OUTPUT_CLOSING_TAG,
        )
    )
    return "\n".join(lines)


def _split_trusted_suffixes(message: dict, content: str) -> tuple[str, str]:
    """Separate authenticated framework suffixes from spillable tool output.

    The marker text alone is not trusted because a tool can emit a lookalike.
    ``agent_runtime_helpers`` sets the private metadata key only when Hermes
    itself appends a genuine user steer. Keeping the suffix outside persisted
    previews ensures aggregate compaction can never hide the user's message.
    """

    suffixes = message.get(POST_PERSIST_SUFFIXES_KEY)
    trusted_parts = (
        [item for item in suffixes if isinstance(item, str) and item]
        if isinstance(suffixes, list)
        else []
    )
    oob = message.get(OOB_USER_MESSAGE_KEY)
    if isinstance(oob, str) and oob:
        trusted_parts.append(oob)
    trusted = "".join(trusted_parts)
    if trusted and content.endswith(trusted):
        return content[: -len(trusted)], trusted
    return content, ""


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
    metadata_out: dict | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{_safe_result_filename(tool_use_id)}"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                if metadata_out is not None:
                    metadata_out.clear()
                    metadata_out.update(
                        {"path": remote_path, "original_size": len(content)}
                    )
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if _trusted_persistence_metadata(msg) is None:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        spillable_content, trusted_suffix = _split_trusted_suffixes(msg, content)
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        persisted_metadata: dict = {}
        replacement = maybe_persist_tool_result(
            content=spillable_content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
            metadata_out=persisted_metadata,
        )
        if trusted_suffix:
            replacement += trusted_suffix
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            if persisted_metadata:
                tool_messages[idx][PERSISTED_OUTPUT_METADATA_KEY] = persisted_metadata
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    # Several results may already have been individually persisted before this
    # aggregate pass. Their previews can still add up past a stricter transport
    # budget, so compact the largest previews while preserving every file path.
    if total_size > config.turn_budget:
        persisted = sorted(
            (
                (i, len(str(msg.get("content", ""))))
                for i, msg in enumerate(tool_messages)
                if _trusted_persistence_metadata(msg) is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx, size in persisted:
            if total_size <= config.turn_budget:
                break
            content = str(tool_messages[idx].get("content", ""))
            compactable_content, trusted_suffix = _split_trusted_suffixes(
                tool_messages[idx], content
            )
            metadata = _trusted_persistence_metadata(tool_messages[idx])
            if metadata is None:
                continue
            replacement = _compact_persisted_message(
                compactable_content, metadata
            )
            if trusted_suffix:
                replacement += trusted_suffix
            if replacement == content:
                continue
            tool_messages[idx]["content"] = replacement
            total_size -= size
            total_size += len(replacement)
            logger.info(
                "Budget enforcement: compacted persisted preview %s (%d -> %d chars)",
                tool_messages[idx].get("tool_call_id", idx),
                size,
                len(replacement),
            )

    if total_size > config.turn_budget:
        logger.warning(
            "Tool-result aggregate remains over budget after persistence: %d > %d chars",
            total_size,
            config.turn_budget,
        )

    return tool_messages
