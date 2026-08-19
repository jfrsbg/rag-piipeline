"""
A queue made of directories: one JSON file per message, claimed by atomic
rename between ready/, inflight/, dead/ and tmp/ slots. Requires a filesystem
with atomic rename; scans ready/ per receive, so deep queues get slow.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .base import DEFAULT_VISIBILITY_TIMEOUT, Message, Queue

log = logging.getLogger(__name__)

SLOTS = ("ready", "inflight", "dead", "tmp")


class FileQueue(Queue):
    def __init__(
        self,
        root: Path | str,
        name: str = "default",
        *,
        visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    ):
        self.name = name
        self.visibility_timeout = visibility_timeout
        self.root = Path(root) / name
        for slot in SLOTS:
            (self.root / slot).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"FileQueue({str(self.root)!r})"

    # ------------------------------------------------------------ transport

    def publish(self, body: Mapping[str, Any]) -> str:
        # Nanosecond prefix so a lexicographic sort of the directory is a FIFO
        # order, and a uuid suffix because two publishers can share a tick.
        message_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        envelope = {
            "id": message_id,
            "attempts": 0,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
            "body": dict(body),
        }
        staged = self.root / "tmp" / f"{message_id}.json"
        staged.write_text(json.dumps(envelope, indent=2))
        # Only now is it visible, and it appears whole.
        staged.rename(self._slot("ready", message_id))
        return message_id

    def receive(self) -> Message | None:
        self._reap()
        for candidate in sorted((self.root / "ready").iterdir()):
            if candidate.suffix != ".json":
                continue
            message_id = candidate.stem
            claimed = self._slot("inflight", message_id)
            try:
                # The whole concurrency story, in one syscall: whoever renames
                # it first owns it, and everyone else gets ENOENT and moves on.
                candidate.rename(claimed)
            except (FileNotFoundError, OSError):
                continue

            envelope = self._load(claimed)
            if envelope is None:
                continue
            envelope["attempts"] = int(envelope.get("attempts", 0)) + 1
            claimed.write_text(json.dumps(envelope, indent=2))
            return Message(
                body=envelope["body"],
                receipt=message_id,
                attempts=envelope["attempts"],
                queue=self.name,
            )
        return None

    def ack(self, message: Message) -> None:
        self._slot("inflight", message.receipt).unlink(missing_ok=True)

    def nack(self, message: Message) -> None:
        claimed = self._slot("inflight", message.receipt)
        if not claimed.exists():
            return                                   # already reaped; it is back
        # Attempts are persisted before the message becomes visible again, so
        # the count a retrying consumer sees is the real one.
        envelope = self._load(claimed) or {"body": message.body}
        envelope["attempts"] = message.attempts
        claimed.write_text(json.dumps(envelope, indent=2))
        claimed.rename(self._slot("ready", message.receipt))

    def dead_letter(self, message: Message, reason: str) -> None:
        claimed = self._slot("inflight", message.receipt)
        envelope = self._load(claimed) or {"body": message.body}
        envelope["attempts"] = message.attempts
        envelope["failed_at"] = datetime.now(timezone.utc).isoformat()
        envelope["reason"] = reason
        self._slot("dead", message.receipt).write_text(json.dumps(envelope, indent=2))
        claimed.unlink(missing_ok=True)

    def depth(self) -> int:
        self._reap()
        return sum(1 for p in (self.root / "ready").iterdir() if p.suffix == ".json")

    # ------------------------------------------------------------ inspection

    def dead_count(self) -> int:
        return sum(1 for p in (self.root / "dead").iterdir() if p.suffix == ".json")

    def dead_bodies(self) -> list[dict[str, Any]]:
        return [
            envelope
            for path in sorted((self.root / "dead").iterdir())
            if path.suffix == ".json"
            for envelope in [self._load(path)]
            if envelope is not None
        ]

    def purge(self) -> None:
        """Empty every slot."""
        for slot in SLOTS:
            for path in (self.root / slot).iterdir():
                path.unlink(missing_ok=True)

    # --------------------------------------------------------------- private

    def _slot(self, slot: str, message_id: str) -> Path:
        return self.root / slot / f"{message_id}.json"

    def _load(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            log.warning("%s: unreadable message %s (%s)", self.name, path.name, exc)
            return None

    def _reap(self) -> None:
        """Make claims older than the visibility timeout visible again."""
        cutoff = time.time() - self.visibility_timeout
        for claimed in (self.root / "inflight").iterdir():
            if claimed.suffix != ".json":
                continue
            try:
                if claimed.stat().st_mtime > cutoff:
                    continue
                message_id = claimed.stem
                claimed.rename(self._slot("ready", message_id))
            except (FileNotFoundError, OSError):
                continue                              # another worker reaped it
            log.warning("%s: reclaimed %s after timeout", self.name, message_id)
