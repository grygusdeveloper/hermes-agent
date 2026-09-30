"""Private metadata-only Claude resume checkpoints, scoped to one profile and agent.

Adapted from the clean fork's _load/_save_durable_state and prefix compatibility
contract (9992a59bb6). Unlike the old source, no persistent system-prompt files
or global home cache are used. A published UUID fences failed CLI transcript tails.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from pathlib import Path

from agent.claude_code_protocol import canonical, digest


def valid_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Claude CLI did not supply a checkpoint UUID")
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise RuntimeError("Claude CLI supplied an invalid checkpoint UUID") from exc
    return value


class Checkpoint:
    def __init__(self, home: Path, owner: str):
        directory = home / "claude-code" / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / (digest(owner) + ".json")
        self.lock = os.open(self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.lock)
            self.lock = None
            raise RuntimeError("Claude native conversation already has an active request") from None

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        value = json.loads(self.path.read_text())
        if value.get("version") != 1:
            return None
        valid_uuid(value.get("session_id"))
        valid_uuid(value.get("checkpoint"))
        if not isinstance(value.get("fingerprints"), list):
            raise RuntimeError("Invalid Claude native checkpoint fingerprints")
        return value

    def publish(self, identity: str, session_id: str, checkpoint: str, fingerprints: list[str]) -> None:
        value = {"version": 1, "identity": identity, "session_id": valid_uuid(session_id),
                 "checkpoint": valid_uuid(checkpoint), "fingerprints": fingerprints}
        fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(canonical(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def close(self) -> None:
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None
