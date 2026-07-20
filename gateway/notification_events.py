"""Safe post-delivery event contexts for outbound notification hooks.

This module deliberately knows nothing about Bark or any other notification
provider.  It exposes platform-neutral successful-delivery events while
building a Discord universal link when the source contains valid snowflake
identifiers.  Raw adapter responses and credentials never enter hook context.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 240
_DISCORD_ID_RE = re.compile(r"^[0-9]+$")


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def _valid_discord_id(value: Any) -> bool:
    return bool(value is not None and _DISCORD_ID_RE.fullmatch(str(value)))


def discord_message_url(source: Any, message_id: Any) -> Optional[str]:
    """Return an exact Discord universal link, or ``None`` when unsafe.

    Discord DMs use the documented ``@me`` scope.  Guild/channel/thread links
    use ``scope_id`` (with the temporary legacy ``guild_id`` fallback).  Every
    interpolated ID must be digits-only so event data can never alter the URL
    path structure.
    """
    if _platform_name(getattr(source, "platform", None)) != "discord":
        return None

    channel_id = getattr(source, "thread_id", None) or getattr(source, "chat_id", None)
    if not (_valid_discord_id(channel_id) and _valid_discord_id(message_id)):
        return None

    scope_id = getattr(source, "scope_id", None) or getattr(source, "guild_id", None)
    chat_type = str(getattr(source, "chat_type", "") or "").lower()
    if chat_type == "dm" and not scope_id:
        scope = "@me"
    elif _valid_discord_id(scope_id):
        scope = str(scope_id)
    else:
        return None

    return (
        f"https://discord.com/channels/{scope}/{channel_id}/{message_id}"
    )


def _safe_preview(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    text = str(value or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception:
        # If the central redactor is unavailable, omit content instead of
        # risking an outbound secret leak.
        logger.debug("Delivery-event preview redaction failed", exc_info=True)
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def build_delivery_context(
    *,
    source: Any,
    message_id: Any,
    kind: str,
    preview: Any = "",
    session_key: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a bounded, secret-redacted event context from a successful send."""
    platform = _platform_name(getattr(source, "platform", None))
    chat_id = str(getattr(source, "chat_id", "") or "")
    thread_id = getattr(source, "thread_id", None)
    effective_channel = str(thread_id or chat_id)
    message_id_text = str(message_id or "")
    safe_kind = str(kind or "delivery")[:64]

    context: dict[str, Any] = {
        "event_id": f"{safe_kind}:{platform}:{effective_channel}:{message_id_text}",
        "kind": safe_kind,
        "platform": platform,
        "chat_id": chat_id,
        "chat_type": str(getattr(source, "chat_type", "") or ""),
        "message_id": message_id_text,
        "message_url": discord_message_url(source, message_id_text),
        "preview": _safe_preview(preview),
    }

    optional = {
        "user_id": getattr(source, "user_id", None),
        "thread_id": thread_id,
        "scope_id": (
            getattr(source, "scope_id", None)
            or getattr(source, "guild_id", None)
        ),
        "session_key": session_key,
        "session_id": session_id,
    }
    for key, value in optional.items():
        if value is not None and str(value) != "":
            context[key] = str(value)

    return context


async def emit_delivery_event(
    *,
    hooks: Any,
    event_type: str,
    source: Any,
    message_id: Any,
    kind: str,
    preview: Any = "",
    session_key: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Emit a successful-delivery event without affecting user delivery.

    Notification hooks are strictly best-effort.  A missing or broken hook
    registry never converts an already-successful platform send into a failed
    gateway turn.
    """
    emit = getattr(hooks, "emit", None)
    if not callable(emit):
        return None

    context = build_delivery_context(
        source=source,
        message_id=message_id,
        kind=kind,
        preview=preview,
        session_key=session_key,
        session_id=session_id,
    )
    try:
        result = emit(event_type, context)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning(
            "Post-delivery hook %s failed for %s",
            event_type,
            context.get("event_id"),
            exc_info=True,
        )
    return context
