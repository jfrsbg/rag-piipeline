"""
In-process queue. One interpreter, no durability, no cleanup.

This is the backend for tests and for a single-process run that wants the
worker code path without a broker. It is deliberately the simplest thing that
satisfies the contract in `base`, so a test failure here is a bug in the test,
not in the queue.

Threads share it safely; processes do not see each other at all. The moment you
want two containers, use FileQueue.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from typing import Any, Mapping

from .base import DEFAULT_VISIBILITY_TIMEOUT, Message, Queue


class InMemoryQueue(Queue):
    def __init__(
        self,
        name: str = "memory",
        *,
        visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    ):
        self.name = name
        self.visibility_timeout = visibility_timeout
        self._ready: deque[tuple[str, dict[str, Any], int]] = deque()
        self._inflight: dict[str, tuple[dict[str, Any], int, float]] = {}
        self._dead: list[tuple[dict[str, Any], str]] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def publish(self, body: Mapping[str, Any]) -> str:
        with self._lock:
            message_id = f"{self.name}-{next(self._ids)}"
            self._ready.append((message_id, dict(body), 0))
            return message_id

    def receive(self) -> Message | None:
        with self._lock:
            self._reap()
            if not self._ready:
                return None
            message_id, body, attempts = self._ready.popleft()
            attempts += 1
            self._inflight[message_id] = (body, attempts, time.monotonic())
            return Message(
                body=body, receipt=message_id, attempts=attempts, queue=self.name
            )

    def ack(self, message: Message) -> None:
        with self._lock:
            self._inflight.pop(message.receipt, None)

    def nack(self, message: Message) -> None:
        with self._lock:
            entry = self._inflight.pop(message.receipt, None)
            if entry is None:
                return
            body, attempts, _claimed = entry
            self._ready.append((message.receipt, body, attempts))

    def dead_letter(self, message: Message, reason: str) -> None:
        with self._lock:
            self._inflight.pop(message.receipt, None)
            self._dead.append((message.body, reason))

    def depth(self) -> int:
        with self._lock:
            self._reap()
            return len(self._ready)

    # ------------------------------------------------------------ inspection

    @property
    def dead(self) -> list[tuple[dict[str, Any], str]]:
        """Dead-lettered messages with the reason each one was parked."""
        with self._lock:
            return list(self._dead)

    @property
    def inflight(self) -> int:
        with self._lock:
            return len(self._inflight)

    def _reap(self) -> None:
        """Return claims whose deadline passed. Caller holds the lock."""
        now = time.monotonic()
        expired = [
            receipt
            for receipt, (_b, _a, claimed) in self._inflight.items()
            if now - claimed >= self.visibility_timeout
        ]
        for receipt in expired:
            body, attempts, _claimed = self._inflight.pop(receipt)
            self._ready.append((receipt, body, attempts))


_REGISTRY: dict[str, InMemoryQueue] = {}
_REGISTRY_LOCK = threading.Lock()


def shared(name: str, **kwargs: Any) -> InMemoryQueue:
    """The one queue with this name in this interpreter.

    `open_queue("memory://", "to-parse")` has to return the same object every
    call or a producer and a consumer in one process would talk past each
    other. Directories give FileQueue that for free; memory needs a registry.
    """
    with _REGISTRY_LOCK:
        queue = _REGISTRY.get(name)
        if queue is None:
            queue = _REGISTRY[name] = InMemoryQueue(name, **kwargs)
        return queue


def reset() -> None:
    """Drop every shared queue. Test isolation, nothing else."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
