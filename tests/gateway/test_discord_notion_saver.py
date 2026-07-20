"""Tests for the Discord Notion save helper (real REST shaping via MockTransport)."""

import json

import httpx
import pytest

from plugins.platforms.discord.notion_saver import (
    NotionSaveError,
    markdown_to_blocks,
    derive_title,
    build_page_blocks,
    save_response_to_notion,
)


def test_markdown_to_blocks_basic():
    md = "# Title\n\nA paragraph.\n\n- one\n- two\n\n```python\nprint(1)\n```"
    blocks = markdown_to_blocks(md)
    types = [b["type"] for b in blocks]
    assert "heading_1" in types
    assert "paragraph" in types
    assert types.count("bulleted_list_item") == 2
    assert "code" in types


def test_markdown_code_languages_are_normalized_for_notion():
    md = (
        "```text\nplain\n```\n"
        "```txt\nalso plain\n```\n"
        "```python\nprint(1)\n```\n"
        "```made-up-language\nopaque\n```"
    )
    blocks = markdown_to_blocks(md)
    languages = [block["code"]["language"] for block in blocks if block["type"] == "code"]
    assert languages == ["plain text", "plain text", "python", "plain text"]


def test_rich_text_splits_long_content():
    long = "x" * 4200
    blocks = markdown_to_blocks(long)
    runs = blocks[0]["paragraph"]["rich_text"]
    assert len(runs) >= 3  # split at <=1900 each
    assert all(len(r["text"]["content"]) <= 2000 for r in runs)


def test_derive_title_prefers_prompt():
    assert derive_title("My prompt line\nmore", "# Heading") == "My prompt line"
    assert derive_title("", "# Response Heading\nbody") == "Response Heading"
    assert derive_title(None, "plain body text") == "plain body text"


def test_build_page_blocks_has_source_prompt_response():
    blocks = build_page_blocks(
        prompt="the prompt", response="the response", source_url="https://jump/1"
    )
    dumped = json.dumps(blocks)
    assert "https://jump/1" in dumped
    assert "Prompt" in dumped
    assert "Response" in dumped


@pytest.mark.asyncio
async def test_notion_save_success():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"id": "page-123", "url": "https://notion.so/page-123"}
        )

    url = await save_response_to_notion(
        api_key="secret-key",
        parent_page_id="parent-abc",
        prompt="hi",
        response="# Answer\n\nbody",
        source_url="https://jump/1",
        transport=httpx.MockTransport(handler),
    )
    assert url == "https://notion.so/page-123"
    assert captured["url"].endswith("/v1/pages")
    assert captured["headers"]["notion-version"] == "2025-09-03"
    assert captured["body"]["parent"] == {"page_id": "parent-abc"}
    # title derived from prompt
    title_txt = captured["body"]["properties"]["title"]["title"][0]["text"]["content"]
    assert title_txt == "hi"


@pytest.mark.asyncio
async def test_notion_save_batches_when_over_100_blocks():
    posts = []
    patches = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "pg", "url": "https://notion.so/pg"})
        patches.append(json.loads(request.content))
        return httpx.Response(200, json={})

    # 250 paragraph lines -> >100 blocks total (plus headers)
    body = "\n\n".join(f"line {i}" for i in range(250))
    url = await save_response_to_notion(
        api_key="k",
        parent_page_id="p",
        prompt="t",
        response=body,
        transport=httpx.MockTransport(handler),
    )
    assert url == "https://notion.so/pg"
    assert len(posts[0]["children"]) <= 100
    assert patches, "remaining blocks should be appended via PATCH"


@pytest.mark.asyncio
async def test_notion_save_failure_raises_without_leaking_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "secret token was bad"})

    with pytest.raises(NotionSaveError) as exc:
        await save_response_to_notion(
            api_key="k",
            parent_page_id="p",
            prompt="t",
            response="body",
            transport=httpx.MockTransport(handler),
        )
    # The user-facing message carries only the HTTP status, not the raw body.
    assert "secret token" not in str(exc.value)
    assert "401" in str(exc.value)
