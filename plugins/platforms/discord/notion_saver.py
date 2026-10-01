"""Notion REST save helper for the Discord ``📚 Notion`` save control.

Creates a child page under a configured parent page containing the saved
Discord final response: a title derived from the prompt (or the first response
heading), a source jump link, the prompt, and the full response body converted
from basic Markdown into readable Notion blocks.

Design notes:

  * Uses ``httpx`` with the Notion API version ``2025-09-03``.  A ``transport``
    may be injected (``httpx.MockTransport``) so tests exercise the request
    shaping without network access.
  * Never logs credentials or raw Notion error bodies — failures surface a
    generic message with the HTTP status only.
  * Notion caps a page-create at 100 child blocks and caps a rich-text string
    at 2000 chars.  We split long text at <=1900 chars, send at most 100 blocks
    in the initial create, then append the remainder in batches of 100 via
    ``PATCH /blocks/{id}/children``.
  * Idempotency (repeated click → existing page) is handled by the caller via
    the SQLite store's cached ``notion_page_url``; this module only creates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NOTION_VERSION = "2025-09-03"
NOTION_BASE_URL = "https://api.notion.com/v1"

# Notion hard limits.
_RICH_TEXT_LIMIT = 2000
_RICH_TEXT_SPLIT = 1900
_MAX_INITIAL_CHILDREN = 100
_APPEND_BATCH = 100

# Notion validates code-block languages against a closed enum. Markdown fence
# labels are ecosystem aliases rather than that enum (for example ``text`` and
# ``txt`` versus Notion's ``plain text``), so passing them through can reject
# the entire page. Keep the documented values explicit and degrade unknown
# labels to plain text instead of failing an otherwise valid save.
_NOTION_CODE_LANGUAGES = frozenset(
    {
        "abap",
        "abc",
        "agda",
        "arduino",
        "ascii art",
        "assembly",
        "bash",
        "basic",
        "bnf",
        "c",
        "c#",
        "c++",
        "clojure",
        "coffeescript",
        "coq",
        "css",
        "dart",
        "dhall",
        "diff",
        "docker",
        "ebnf",
        "elixir",
        "elm",
        "erlang",
        "f#",
        "flow",
        "fortran",
        "gherkin",
        "glsl",
        "go",
        "graphql",
        "groovy",
        "haskell",
        "hcl",
        "html",
        "idris",
        "java",
        "java/c/c++/c#",
        "javascript",
        "json",
        "julia",
        "kotlin",
        "latex",
        "less",
        "lisp",
        "livescript",
        "llvm ir",
        "lua",
        "makefile",
        "markdown",
        "markup",
        "mathematica",
        "matlab",
        "mermaid",
        "nix",
        "notion formula",
        "objective-c",
        "ocaml",
        "pascal",
        "perl",
        "php",
        "plain text",
        "powershell",
        "prolog",
        "protobuf",
        "purescript",
        "python",
        "r",
        "racket",
        "reason",
        "ruby",
        "rust",
        "sass",
        "scala",
        "scheme",
        "scss",
        "shell",
        "smalltalk",
        "solidity",
        "sql",
        "swift",
        "toml",
        "typescript",
        "vb.net",
        "verilog",
        "vhdl",
        "visual basic",
        "webassembly",
        "xml",
        "yaml",
    }
)
_CODE_LANGUAGE_ALIASES = {
    "csharp": "c#",
    "cs": "c#",
    "cxx": "c++",
    "cpp": "c++",
    "dockerfile": "docker",
    "fish": "shell",
    "js": "javascript",
    "md": "markdown",
    "objectivec": "objective-c",
    "objc": "objective-c",
    "plain": "plain text",
    "plaintext": "plain text",
    "ps1": "powershell",
    "pwsh": "powershell",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "sh": "shell",
    "text": "plain text",
    "ts": "typescript",
    "txt": "plain text",
    "yml": "yaml",
    "zsh": "shell",
}


class NotionSaveError(Exception):
    """Raised when a Notion save fails.  Message is safe to show a user."""


def _rich_text(content: str) -> List[Dict[str, Any]]:
    """Split *content* into <=1900-char rich-text runs (<=2000 hard cap)."""
    content = content or ""
    if not content:
        return [{"type": "text", "text": {"content": ""}}]
    runs: List[Dict[str, Any]] = []
    remaining = content
    while remaining:
        chunk = remaining[:_RICH_TEXT_SPLIT]
        remaining = remaining[_RICH_TEXT_SPLIT:]
        runs.append({"type": "text", "text": {"content": chunk}})
    return runs


def _paragraph(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _heading(text: str, level: int) -> Dict[str, Any]:
    level = max(1, min(level, 3))
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": _rich_text(text)},
    }


def _bullet(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text)},
    }


def _numbered(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": _rich_text(text)},
    }


def _normalize_code_language(language: str) -> str:
    normalized = (language or "plain text").strip().lower()
    normalized = _CODE_LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in _NOTION_CODE_LANGUAGES:
        return "plain text"
    return normalized


def _code(text: str, language: str = "plain text") -> Dict[str, Any]:
    # Notion code blocks accept only a single rich-text run per split; keep
    # each run within the rich-text cap.
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": _rich_text(text),
            "language": _normalize_code_language(language),
        },
    }


def markdown_to_blocks(text: str) -> List[Dict[str, Any]]:
    """Convert basic Markdown to Notion blocks (headings/lists/code/paragraphs)."""
    blocks: List[Dict[str, Any]] = []
    lines = (text or "").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code blocks.
        if stripped.startswith("```"):
            language = stripped[3:].strip() or "plain text"
            i += 1
            code_lines: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            blocks.append(_code("\n".join(code_lines), language))
            continue

        if not stripped:
            i += 1
            continue

        # Headings.
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            content = stripped[hashes:].strip()
            blocks.append(_heading(content, hashes))
            i += 1
            continue

        # Bulleted list.
        if stripped[:2] in ("- ", "* ") or stripped[:2] == "+ ":
            blocks.append(_bullet(stripped[2:].strip()))
            i += 1
            continue

        # Numbered list ("1. ", "2) ").
        num = _numbered_prefix(stripped)
        if num is not None:
            blocks.append(_numbered(num))
            i += 1
            continue

        # Paragraph — greedily join consecutive plain lines.
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("```")
                or nxt[:2] in ("- ", "* ", "+ ")
                or _numbered_prefix(nxt) is not None
            ):
                break
            para_lines.append(nxt)
            i += 1
        blocks.append(_paragraph(" ".join(para_lines)))
    return blocks


def _numbered_prefix(line: str) -> Optional[str]:
    """Return list-item text if *line* starts with ``N.``/``N)``, else None."""
    j = 0
    while j < len(line) and line[j].isdigit():
        j += 1
    if j == 0 or j >= len(line):
        return None
    if line[j] in ".)" and j + 1 < len(line) and line[j + 1] == " ":
        return line[j + 2:].strip()
    return None


def _first_heading(text: str) -> Optional[str]:
    for line in (text or "").split("\n"):
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or None
        if s:
            return None
    return None


def derive_title(prompt: Optional[str], response: str) -> str:
    """Title from the prompt's first line, falling back to a response heading."""
    prompt = (prompt or "").strip()
    if prompt:
        first = prompt.split("\n", 1)[0].strip()
        if first:
            return first[:180]
    heading = _first_heading(response)
    if heading:
        return heading[:180]
    body = (response or "").strip().split("\n", 1)[0].strip()
    return (body[:180] or "Hermes response")


def build_page_blocks(
    *, prompt: Optional[str], response: str, source_url: Optional[str]
) -> List[Dict[str, Any]]:
    """Assemble the full ordered block list for the saved-response page."""
    blocks: List[Dict[str, Any]] = []
    if source_url:
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "Source: ", },
                            "annotations": {"bold": True},
                        },
                        {
                            "type": "text",
                            "text": {"content": source_url, "link": {"url": source_url}},
                        },
                    ]
                },
            }
        )
    if prompt and prompt.strip():
        blocks.append(_heading("Prompt", 2))
        blocks.extend(markdown_to_blocks(prompt))
    blocks.append(_heading("Response", 2))
    blocks.extend(markdown_to_blocks(response))
    return blocks


async def save_response_to_notion(
    *,
    api_key: str,
    parent_page_id: str,
    prompt: Optional[str],
    response: str,
    source_url: Optional[str] = None,
    transport: Any = None,
    timeout: float = 30.0,
) -> str:
    """Create a Notion child page for the response and return its URL.

    Raises :class:`NotionSaveError` on failure (message safe to surface).
    """
    import httpx

    if not api_key:
        raise NotionSaveError("Notion is not configured (missing API key).")
    if not parent_page_id:
        raise NotionSaveError("Notion is not configured (missing parent page).")

    title = derive_title(prompt, response)
    all_blocks = build_page_blocks(
        prompt=prompt, response=response, source_url=source_url
    )
    initial = all_blocks[:_MAX_INITIAL_CHILDREN]
    remaining = all_blocks[_MAX_INITIAL_CHILDREN:]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        },
        "children": initial,
    }

    client_kwargs: Dict[str, Any] = {"timeout": timeout, "headers": headers}
    if transport is not None:
        client_kwargs["transport"] = transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            resp = await client.post(f"{NOTION_BASE_URL}/pages", json=payload)
        except Exception:
            # Never log the exception body — it can echo request headers.
            raise NotionSaveError("Could not reach Notion.")
        if resp.status_code not in (200, 201):
            # Log only the status; never the raw error body (may leak context).
            logger.warning("Notion page create failed: HTTP %s", resp.status_code)
            raise NotionSaveError(f"Notion save failed (HTTP {resp.status_code}).")

        data = resp.json()
        page_id = data.get("id")
        page_url = data.get("url") or ""

        # Append any blocks beyond the initial 100 in batches.
        idx = 0
        while remaining and page_id:
            batch = remaining[idx : idx + _APPEND_BATCH]
            if not batch:
                break
            try:
                append_resp = await client.patch(
                    f"{NOTION_BASE_URL}/blocks/{page_id}/children",
                    json={"children": batch},
                )
            except Exception:
                raise NotionSaveError("Could not reach Notion while appending content.")
            if append_resp.status_code not in (200, 201):
                logger.warning(
                    "Notion append failed: HTTP %s", append_resp.status_code
                )
                raise NotionSaveError(
                    f"Notion save incomplete (HTTP {append_resp.status_code})."
                )
            idx += _APPEND_BATCH
            if idx >= len(remaining):
                break

    return page_url
