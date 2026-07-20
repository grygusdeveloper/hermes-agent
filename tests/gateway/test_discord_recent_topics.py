"""Tests for the Discord /topics recent-response index."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.commands import (
    ACTIVE_SESSION_BYPASS_COMMANDS,
    resolve_command,
    slack_native_slashes,
    telegram_bot_commands,
)
from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _collect_recent_response_topics,
    _discord_topic_snippet,
    _looks_like_nonconversational_history_message,
    _redact_discord_topic_text,
)


class _HistoryChannel:
    def __init__(self, messages, channel_id=77):
        self.id = channel_id
        self.guild = None
        self._messages = list(messages)
        self.history_kwargs = None

    def history(self, **kwargs):
        self.history_kwargs = kwargs

        async def _iterate():
            for message in self._messages:
                yield message

        return _iterate()


def _author(author_id, *, bot=False):
    return SimpleNamespace(id=author_id, bot=bot)


def _message(message_id, author, content, *, channel=None, jump_url=None):
    channel = channel or SimpleNamespace(id=77, guild=None)
    values = {
        "id": message_id,
        "author": author,
        "content": content,
        "clean_content": content,
        "attachments": [],
        "channel": channel,
        "guild": getattr(channel, "guild", None),
    }
    if jump_url is not None:
        values["jump_url"] = jump_url
    return SimpleNamespace(**values)


def _is_nonconversational(message):
    return _looks_like_nonconversational_history_message(message.content)


def test_collect_recent_response_topics_groups_chunks_and_skips_control_traffic():
    bot = _author(999, bot=True)
    other_bot = _author(555, bot=True)
    user = _author(42)

    chronological = [
        _message(1, user, "<@999> Explain deterministic provider routing in detail"),
        _message(
            15,
            bot,
            "◐ Session automatically reset (inactive). Conversation history cleared.",
        ),
        _message(2, bot, "First answer chunk", jump_url="https://discord.test/2"),
        _message(3, bot, "Second answer chunk", jump_url="https://discord.test/3"),
        _message(4, user, "/status"),
        _message(5, bot, "Session status", jump_url="https://discord.test/5"),
        _message(6, user, "Build a topic index for this long chat"),
        _message(7, user, "Make every item link to the first response message"),
        _message(8, other_bot, "Unrelated bot chatter"),
        _message(9, bot, "⏳ Working — 2 min", jump_url="https://discord.test/9"),
        _message(
            95,
            bot,
            "⚠️ Gateway shutting down — Your current task will be interrupted.",
            jump_url="https://discord.test/95",
        ),
        _message(
            96,
            bot,
            "⚠️ Iteration budget exhausted (90/90) — asking model to summarise",
            jump_url="https://discord.test/96",
        ),
        _message(10, bot, "Final answer", jump_url="https://discord.test/10"),
        _message(11, bot, "Final answer continuation", jump_url="https://discord.test/11"),
        _message(12, bot, "## Recent answered topics\n1. control output"),
    ]

    topics = _collect_recent_response_topics(
        list(reversed(chronological)),
        bot_user=bot,
        limit=10,
        is_nonconversational=_is_nonconversational,
    )

    assert [topic["title"] for topic in topics] == [
        "Build a topic index for this long chat",
        "Explain deterministic provider routing in detail",
    ]
    assert [topic["response_message_id"] for topic in topics] == ["10", "2"]
    assert [topic["jump_url"] for topic in topics] == [
        "https://discord.test/10",
        "https://discord.test/2",
    ]


def test_collect_recent_response_topics_links_to_final_answer_block_not_work_notes():
    bot = _author(999, bot=True)
    user = _author(42)
    nonconversational_ids = {3, 5, 9}
    chronological = [
        _message(1, user, "Audit this project deeply"),
        _message(2, bot, "I'll inspect the repository beyond its README."),
        _message(3, bot, "📚 Reading repository files"),
        _message(4, bot, "Initial signal: the architecture needs more review."),
        _message(5, bot, "🔎 Searching dependency advisories"),
        _message(6, bot, "One final implementation detail remains to check."),
        _message(7, bot, "# Complete assessment\n\n## Executive verdict"),
        _message(8, bot, "Final assessment continuation"),
        _message(9, bot, "💾 Self-improvement review: skill updated"),
    ]

    topics = _collect_recent_response_topics(
        list(reversed(chronological)),
        bot_user=bot,
        limit=10,
        is_nonconversational=lambda message: message.id in nonconversational_ids,
        is_final_response=lambda message: message.id == 7,
    )

    assert topics[0]["response_message_id"] == "7"
    assert topics[0]["jump_url"] == "https://discord.com/channels/@me/77/7"


def test_collect_recent_response_topics_recognizes_legacy_split_final_answer():
    bot = _author(999, bot=True)
    user = _author(42)
    chronological = [
        _message(1, user, "Review the repository"),
        _message(2, bot, "I'll check one final supply-chain detail."),
        _message(3, bot, "## Final verdict\n\nPilot it carefully. (1/2)"),
        _message(4, bot, "## Sources\n\nVerified references. (2/2)"),
    ]

    topics = _collect_recent_response_topics(
        list(reversed(chronological)),
        bot_user=bot,
        limit=10,
    )

    assert topics[0]["response_message_id"] == "3"


def test_collect_recent_response_topics_limits_newest_and_builds_dm_jump_url():
    bot = _author(999, bot=True)
    user = _author(42)
    channel = SimpleNamespace(id=77, guild=None)
    chronological = [
        _message(1, user, "Older topic", channel=channel),
        _message(2, bot, "Older answer", channel=channel),
        _message(3, user, "Newest topic", channel=channel),
        _message(4, bot, "Newest answer", channel=channel),
    ]

    topics = _collect_recent_response_topics(
        list(reversed(chronological)),
        bot_user=bot,
        limit=1,
    )

    assert topics == [
        {
            "title": "Newest topic",
            "jump_url": "https://discord.com/channels/@me/77/4",
            "prompt_message_id": "3",
            "response_message_id": "4",
        }
    ]


def test_topic_titles_do_not_echo_credentials():
    oauth_code = "4/0AX" + "Ab1_" * 12
    api_key = "sk-" + "Ab1c" * 10

    assert _redact_discord_topic_text(oauth_code) == (
        "Sensitive credential/authentication message"
    )
    redacted_prose = _redact_discord_topic_text(f"Use this API key {api_key} for login")
    assert api_key not in redacted_prose
    assert "Ab1c" not in redacted_prose
    assert "[REDACTED]" in redacted_prose


def test_topic_titles_replace_embedded_urls_that_break_discord_jump_links():
    user = _author(42)

    bare_url = _message(
        1,
        user,
        "Analyze this project https://github.com/Egonex-AI/Understand-Anything",
    )
    markdown_url = _message(
        2,
        user,
        "Review [VaultSync](https://github.com/psimaker/vaultsync) for Draw",
    )

    assert _discord_topic_snippet(bare_url) == "Analyze this project (GitHub link)"
    assert _discord_topic_snippet(markdown_url) == "Review VaultSync (GitHub link) for Draw"


@pytest.mark.asyncio
async def test_discord_adapter_scans_before_trigger_and_returns_topics():
    bot = _author(999, bot=True)
    user = _author(42)
    channel = _HistoryChannel(
        [
            _message(2, bot, "Answer", jump_url="https://discord.test/2"),
            _message(1, user, "Question"),
        ]
    )
    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(
        user=bot,
        get_channel=lambda channel_id: channel if channel_id == 77 else None,
        fetch_channel=AsyncMock(return_value=None),
    )
    adapter._nonconversational_messages = set()

    topics = await adapter.get_recent_response_topics(
        "77", before_message_id="500", limit=10
    )

    assert topics[0]["title"] == "Question"
    assert topics[0]["response_message_id"] == "2"
    assert channel.history_kwargs["limit"] == 250
    assert channel.history_kwargs["oldest_first"] is False
    assert channel.history_kwargs["before"].id == 500


class _TopicsRunner(GatewaySlashCommandsMixin):
    def __init__(self, indexer):
        self.adapters = {
            Platform.DISCORD: SimpleNamespace(get_recent_response_topics=indexer)
        }


@pytest.mark.asyncio
async def test_topics_handler_formats_links_and_forwards_count_and_anchor():
    indexer = AsyncMock(
        return_value=[
            {
                "title": "Topic [with brackets]",
                "jump_url": "https://discord.test/answer",
            },
            {
                "title": "Analyze https://github.com/example/project",
                "jump_url": "https://discord.test/answer-2",
            },
        ]
    )
    runner = _TopicsRunner(indexer)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="77",
        user_id="42",
        message_id="500",
    )
    event = MessageEvent(
        text="/topics 2",
        message_type=MessageType.COMMAND,
        source=source,
        message_id="500",
    )

    result = await runner._handle_topics_command(event)

    assert "## Recent answered topics" in result
    assert "[Topic \\[with brackets\\]](https://discord.test/answer)" in result
    assert "[Analyze (GitHub link)](https://discord.test/answer-2)" in result
    assert result.count("https://") == 2
    indexer.assert_awaited_once_with("77", before_message_id="500", limit=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["0", "26", "many"])
async def test_topics_handler_rejects_invalid_counts(arg):
    indexer = AsyncMock(return_value=[])
    runner = _TopicsRunner(indexer)
    event = MessageEvent(
        text=f"/topics {arg}",
        message_type=MessageType.COMMAND,
        source=SessionSource(platform=Platform.DISCORD, chat_id="77"),
    )

    result = await runner._handle_topics_command(event)

    assert result.startswith("Usage: `/topics")
    indexer.assert_not_awaited()


def test_topics_command_is_registered_and_safe_while_agent_is_busy():
    command = resolve_command("topics")

    assert command is not None
    assert command.gateway_only is True
    assert command.args_hint == "[count]"
    assert "topics" in ACTIVE_SESSION_BYPASS_COMMANDS
    assert "topics" not in {name for name, _description in telegram_bot_commands()}
    assert "topics" not in {name for name, _description, _hint in slack_native_slashes()}
