"""Typed Claude Code CLI failures carrying API status and an Anthropic-style body.

Ported from the fork's ClaudeCodeSession error types (7025953856, 33d79181f5,
d6687f3f90, 9112f82f46, 232446c398, 9daf44b209). ``status_code`` and ``body``
(also via ``response.json()``) let Hermes's error classifier route a CLI-reported
failure like a direct API error: 401 auth, 429 rate limit, 500 server, 529
overload. The text helpers only ever look at what the CLI itself wrote (result
events, ``<synthetic>`` banners, stderr), never at the model's own reply.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any


class ClaudeCodeSessionExpired(RuntimeError):
    """The CLI positively identified a missing/expired session or unknown checkpoint."""


class ClaudeCodeLaunchError(RuntimeError):
    """The CLI cannot be started (missing, not executable, unusable working directory).

    Deterministic until the installation is fixed, so never retried. Deliberately
    not an ``OSError``: the classifier treats those as transient transport failures.
    """


class CliFlagRejected(RuntimeError):
    """The CLI rejected an optional flag before reading the request."""


# Anthropic error types by HTTP status, for results that name no type.
_API_ERROR_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
    500: "api_error", 529: "overloaded_error",
}


def api_error_body(status: int | None, code: str | None, message: str) -> dict[str, Any]:
    return {"error": {"type": code or _API_ERROR_TYPES.get(status or 0) or "api_error", "message": message}}


class ClaudeCodeAPIError(RuntimeError):
    """A Claude Code turn failed with an API error the CLI reported."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 body: dict[str, Any] | None = None, detail: str = "") -> None:
        super().__init__(message)
        valid = isinstance(status_code, int) and not isinstance(status_code, bool)
        self.status_code = status_code if valid else None
        self.body = body if isinstance(body, dict) else {}
        self.detail = detail or ""
        payload = self.body
        self.response = SimpleNamespace(status_code=self.status_code, headers={}, json=lambda: payload)


class ClaudeCodeRefusal(ClaudeCodeAPIError):
    """Claude declined to answer (API stop reason ``refusal``, no CLI fallback model).

    Deterministic for the unchanged request; its body type ``refusal`` classifies as
    a content-policy block. ``detail`` is the CLI's explanation.
    """


class ClaudeCodeSoftLimitNotice(ClaudeCodeAPIError):
    """A usage/rate-limit notice instead of an answer: retried, never delivered."""


class ClaudeCodeUsageLimitError(ClaudeCodeAPIError):
    """The subscription limit holds until a reset: HTTP 429, body ``usage_limit_reached``."""

    def __init__(self, message: str, *, body: dict[str, Any], detail: str = "",
                 resets_at: float | None = None, rate_limit_type: str | None = None) -> None:
        super().__init__(message, status_code=429, body=body, detail=detail)
        self.resets_at = resets_at
        self.rate_limit_type = rate_limit_type


# Markers that the CLI's server session (or the resume checkpoint) is gone.
_EXPIRED_SESSION_MARKERS = (
    "session not found", "invalid session", "session expired", "session has expired",
    "unknown session", "no such session", "could not resume", "resume failed",
    "session does not exist", "no conversation found", "conversation not found",
    # --resume-session-at with a checkpoint the loaded chain does not hold.
    "no message found with message.uuid",
)

# Soft notices the CLI may write instead of an answer. Observed live: "You've hit
# your monthly spend limit · raise it at claude.ai/settings/usage".
_SOFT_LIMIT_MARKERS = (
    "monthly spend limit", "claude.ai/settings/usage", "out of extra usage",
    "usage limit reached", "hit your usage limit", "rate limit reached", "too many requests",
    "limit · resets",
)
# "You've hit your limit", "You've hit your session limit · resets 3pm (UTC)".
LIMIT_BANNER_RE = re.compile(r"\bhit your ((?:[\w-]+ ){0,3})limit\b", re.IGNORECASE)
# ...and the wording that says the window only lifts at its reset.
HARD_LIMIT_TEXT_RE = re.compile(
    r"(?:\bhit your (?:[\w-]+ ){0,3}limit|\blimit reached)\s*[·•-]\s*(?:resets\b|continues automatically)",
    re.IGNORECASE,
)


def is_expired_session_error(detail: str) -> bool:
    normalized = (detail or "").lower()
    return any(marker in normalized for marker in _EXPIRED_SESSION_MARKERS)


def is_soft_limit_detail(detail: str) -> bool:
    """A CLI-written error detail that reads as a rate/spend/usage limit.

    The CLI reports these as ``is_error`` results with ``api_error_status`` 429 or
    an ``error: rate_limit`` assistant flag, often with a limit banner.
    """
    if not detail:
        return False
    lower = detail.lower()
    if re.search(r'"error"\s*:\s*"rate_limit"', lower) or re.search(r'api_error_status"\s*:\s*429', lower):
        return True
    return any(marker in lower for marker in _SOFT_LIMIT_MARKERS) or bool(LIMIT_BANNER_RE.search(lower))


def is_soft_limit_notice(text: str) -> bool:
    """A short, banner-like CLI text (a ``<synthetic>`` message) about usage limits."""
    body = (text or "").strip()
    if not body or len(body) > 600:
        return False
    normalized = body.lower()
    if not any(marker in normalized for marker in _SOFT_LIMIT_MARKERS) and not LIMIT_BANNER_RE.search(normalized):
        return False
    return len([line for line in body.splitlines() if line.strip()]) <= 6


def result_error(event: dict[str, Any]) -> tuple[int | None, str | None, str]:
    """``(api_error_status, api_error_code, text)`` of a failed result event."""
    status = event.get("api_error_status")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status < 600:
        status = None
    code = event.get("api_error_code")
    code = code.strip() if isinstance(code, str) and code.strip() else None
    text = event.get("result")
    return status, code, text[:300] if isinstance(text, str) else ""


def refusal_error(text: Any) -> ClaudeCodeRefusal:
    detail = text.strip()[:1000] if isinstance(text, str) and text.strip() else (
        "Claude declined to respond to this request.")
    return ClaudeCodeRefusal(f"Claude Code refusal: {detail}",
                             body={"error": {"type": "refusal", "message": detail}}, detail=detail)


def soft_limit_exhausted(attempts: int, cause: ClaudeCodeAPIError) -> ClaudeCodeAPIError:
    """The error once soft notices used up the retry budget (HTTP 429 unless the CLI said otherwise)."""
    notice = cause.detail or str(cause)
    status = cause.status_code or 429
    body = cause.body or api_error_body(status, None, notice[:400])
    return ClaudeCodeAPIError(
        "Claude Code CLI returned a soft usage/limit notice "
        f"after {attempts} attempts (not treated as an answer). Detail: {notice[:400]}",
        status_code=status, body=body, detail=notice)


def model_fallback_notice(subtype: str, event: dict[str, Any]) -> str:
    """The user-facing line for a CLI model fallback or refusal event (CLI text, never the model's)."""
    content = event.get("content")
    if isinstance(content, str) and content.strip():
        return "⚠️ Claude Code: " + " ".join(content.split())[:300]
    original = event.get("originalModel") or event.get("original_model") or "Claude"
    fallback = event.get("fallbackModel") or event.get("fallback_model") or "another model"
    category = event.get("apiRefusalCategory") or event.get("api_refusal_category")
    why = f" ({category})" if category else ""
    if subtype == "model_refusal_no_fallback":
        return f"⚠️ {original} declined this request{why}."
    if subtype == "model_refusal_fallback":
        return f"⚠️ {original} declined this request{why}; {fallback} answered instead."
    return f"⚠️ {original} was unavailable; {fallback} answered instead."
