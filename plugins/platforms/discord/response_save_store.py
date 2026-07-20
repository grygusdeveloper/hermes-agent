"""Durable SQLite store for Discord final-response save controls.

Backs the ``⭐ Favorite`` / ``📚 Notion`` buttons attached to normal Discord
assistant final responses.  Each stored record is keyed by the *control*
message ID — the first Discord message/chunk of the logical response, which is
the message the persistent View is attached to.  ``interaction.message.id`` on
a button click resolves back to that key, so the callback can load the full
response even after a gateway restart.

Two tables:

  * ``responses``  — one row per saved final response (content, prompt/title,
    channel/guild IDs, jump URL, timestamp, and a cached Notion page URL for
    idempotent Notion saves).
  * ``favorites``  — one row per (response, user) favorite.  The composite
    primary key gives per-response+user uniqueness, so a repeated ⭐ click is a
    no-op rather than a duplicate.

All methods are synchronous and cheap; async callers wrap them in
``asyncio.to_thread`` when running on the event loop.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DiscordResponseSaveStore:
    """Thread-safe SQLite store for saved Discord final responses + favorites."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    message_id   TEXT PRIMARY KEY,
                    content      TEXT NOT NULL,
                    prompt       TEXT,
                    channel_id   TEXT,
                    guild_id     TEXT,
                    jump_url     TEXT,
                    created_ts   REAL NOT NULL,
                    notion_page_url TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS favorites (
                    message_id TEXT NOT NULL,
                    user_id    TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    PRIMARY KEY (message_id, user_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fav_user ON favorites(user_id, created_ts DESC)"
            )

    # ── response records ────────────────────────────────────────────────
    def save_response(
        self,
        *,
        message_id: str,
        content: str,
        prompt: Optional[str] = None,
        channel_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        jump_url: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Insert or update the durable record for a final response.

        Keyed by the first/control ``message_id``.  Re-storing the same key
        (e.g. a finalize edit after the initial send) refreshes the content and
        metadata but preserves any cached Notion page URL so idempotency holds.
        """
        if not message_id:
            return
        ts = float(timestamp) if timestamp is not None else time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO responses
                    (message_id, content, prompt, channel_id, guild_id, jump_url, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    content = excluded.content,
                    prompt = excluded.prompt,
                    channel_id = excluded.channel_id,
                    guild_id = excluded.guild_id,
                    jump_url = excluded.jump_url
                """,
                (
                    str(message_id),
                    content or "",
                    prompt,
                    channel_id,
                    guild_id,
                    jump_url,
                    ts,
                ),
            )

    def get_response(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM responses WHERE message_id = ?", (str(message_id),)
            ).fetchone()
        return dict(row) if row else None

    def set_notion_page(self, message_id: str, page_url: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE responses SET notion_page_url = ? WHERE message_id = ?",
                (page_url, str(message_id)),
            )

    def get_notion_page(self, message_id: str) -> Optional[str]:
        row = self.get_response(message_id)
        if not row:
            return None
        return row.get("notion_page_url") or None

    # ── favorites ───────────────────────────────────────────────────────
    def add_favorite(self, message_id: str, user_id: str) -> bool:
        """Record a favorite; returns True if newly added, False if duplicate."""
        if not message_id or not user_id:
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO favorites (message_id, user_id, created_ts) VALUES (?, ?, ?)",
                (str(message_id), str(user_id), time.time()),
            )
            return cur.rowcount > 0

    def is_favorited(self, message_id: str, user_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM favorites WHERE message_id = ? AND user_id = ?",
                (str(message_id), str(user_id)),
            ).fetchone()
        return row is not None

    def list_favorites(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the newest favorites for a user as joined response records."""
        limit = max(1, min(int(limit), 25))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, f.created_ts AS favorited_ts
                FROM favorites f
                JOIN responses r ON r.message_id = f.message_id
                WHERE f.user_id = ?
                ORDER BY f.created_ts DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
        return [dict(row) for row in rows]
