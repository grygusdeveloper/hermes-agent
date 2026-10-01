"""Pure helpers for rendering safe Discord topic-index link labels."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_MARKDOWN_URL_RE = re.compile(
    r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
    re.IGNORECASE,
)
_BARE_URL_RE = re.compile(r"https?://[^\s<>\])]+", re.IGNORECASE)


def _discord_topic_url_kind(url: str) -> str:
    """Return a non-linking human label for a URL's service."""
    try:
        host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except (TypeError, ValueError):
        host = ""
    if host == "github.com" or host.endswith(".github.com"):
        return "GitHub"
    if host in {"youtu.be", "youtube.com"} or host.endswith(".youtube.com"):
        return "YouTube"
    if host == "discord.com" or host.endswith(".discord.com"):
        return "Discord"
    return "Web"


def _url_replacement(url: str) -> str:
    """Replace a URL without turning the replacement into another auto-link."""
    trailing = ""
    while url and url[-1] in ".,!?;:":
        trailing = url[-1] + trailing
        url = url[:-1]
    return f"({_discord_topic_url_kind(url)} link){trailing}"


def sanitize_discord_topic_label(text: str) -> str:
    """Remove nested URLs that make Discord ignore an outer Markdown link.

    Discord does not reliably render ``[label containing https://…](jump)`` as
    a jump link. Markdown links inside labels are invalid for the same reason.
    Preserve useful anchor text/service context while ensuring the final label
    contains no URL of its own.
    """
    value = str(text or "")
    value = _MARKDOWN_URL_RE.sub(
        lambda match: f"{match.group(1)} {_url_replacement(match.group(2))}",
        value,
    )
    value = _BARE_URL_RE.sub(lambda match: _url_replacement(match.group(0)), value)
    return re.sub(r"\s+", " ", value).strip()
