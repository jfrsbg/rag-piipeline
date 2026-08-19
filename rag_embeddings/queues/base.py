"""
Queue interface: backends supply transport, the retry loop lives here.

Backend contract: a received message stays invisible to other consumers until
acked, nacked, or its visibility deadline passes; `ack` removes it, `nack`
redelivers it with `attempts` incremented, and so does a passed deadline.
Delivery is therefore at-least-once, never exactly-once.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

log = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_VISIBILITY_TIMEOUT = 300.0
DEFAULT_POLL_INTERVAL = 0.5


@dataclass(frozen=True)
class Message:
    """One delivery. `receipt` is the backend's opaque handle."""

    body: dict[str, Any]
    receipt: str
    attempts: int = 1
    queue: str = ""

    @property
    def redelivered(self) -> bool:
        return self.attempts > 1

    @property
    def label(self) -> str:
        """Return a short, greppable identifier for log lines."""
        sha = self.body.get("sha256")
        uri = self.body.get("uri")
        parts = [p for p in (sha[:12] if isinstance(sha, str) else None, uri) if p]
        return f"{self.receipt} ({' '.join(parts)})" if parts else self.receipt


@dataclass
class ConsumeStats:
    """Tally of what one consume run did."""

    received: int = 0
    acked: int = 0
    failed: int = 0
    dead_lettered: int = 0
    errors: list[str] = field(default_factory=list)
    # True when the loop ended because it was signalled rather than because the
    # queue drained. A service exiting is normally the former, and the two are
    # worth telling apart in a log when a container disappears unexpectedly.
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.dead_lettered == 0


class Queue(ABC):
    """A named queue. Subclasses implement transport and nothing else."""

    name: str
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT

    # ------------------------------------------------------------ transport

    @abstractmethod
    def publish(self, body: Mapping[str, Any]) -> str:
        """Enqueue one message. Returns its id."""

    @abstractmethod
    def receive(self) -> Message | None:
        """Claim one message, or None if the queue is empty. Non-blocking."""

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Done with it. The message is gone."""

    @abstractmethod
    def nack(self, message: Message) -> None:
        """Failed. Back on the queue with a bumped attempt count."""

    @abstractmethod
    def dead_letter(self, message: Message, reason: str) -> None:
        """Out of attempts. Park it where a human will find it."""

    @abstractmethod
    def depth(self) -> int:
        """Return the number of messages waiting."""

    def close(self) -> None:
        """Release whatever the backend holds. Default: nothing."""

    # ---------------------------------------------------------------- above

    def publish_all(self, bodies: list[Mapping[str, Any]]) -> list[str]:
        return [self.publish(body) for body in bodies]

    def receive_batch(self, max_messages: int = 10) -> list[Message]:
        """Claim up to `max_messages` at once; override if the backend batches."""
        batch = []
        for _ in range(max(0, max_messages)):
            message = self.receive()
            if message is None:
                break
            batch.append(message)
        return batch

    def poll(
        self,
        *,
        timeout: float | None = None,
        interval: float = DEFAULT_POLL_INTERVAL,
        should_stop: Callable[[], bool] | None = None,
    ) -> Message | None:
        """Receive, waiting up to `timeout` (None waits forever) for a message."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if should_stop is not None and should_stop():
                return None
            message = self.receive()
            if message is not None:
                return message
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(interval)

    def consume(
        self,
        handler: Callable[[dict[str, Any]], Any],
        *,
        max_messages: int | None = None,
        idle_timeout: float | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        on_error: Callable[[Message, Exception], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> ConsumeStats:
        """Run `handler` over messages, retrying until `max_attempts` then dead-lettering."""
        stats = ConsumeStats()

        while max_messages is None or stats.received < max_messages:
            message = self.poll(
                timeout=idle_timeout,
                interval=poll_interval,
                should_stop=should_stop,
            )
            if message is None:
                # Either the queue stayed quiet for `idle_timeout`, or the poll
                # was cut short by a signal while it waited.
                stats.stopped = should_stop is not None and should_stop()
                break

            stats.received += 1
            if message.redelivered:
                log.info(
                    "%s: redelivery %d of %s", self.name, message.attempts,
                    message.receipt,
                )

            # One line on the way in and one on the way out, both carrying the
            # same label: a document that never reaches its "done" line is
            # either still in the handler or died there, and the pair is what
            # makes that distinction visible in a container log.
            log.info("%s: received %s", self.name, message.label)
            started = time.monotonic()
            try:
                handler(message.body)
            except Exception as exc:                    # noqa: BLE001
                log.info(
                    "%s: failed %s after %.1fs",
                    self.name, message.label, time.monotonic() - started,
                )
                stats.failed += 1
                stats.errors.append(f"{message.receipt}: {exc}")
                if on_error is not None:
                    on_error(message, exc)

                if message.attempts >= max_attempts:
                    stats.dead_lettered += 1
                    log.error(
                        "%s: dead-lettering after %d attempts: %s",
                        self.name, message.attempts, exc,
                    )
                    self.dead_letter(message, str(exc))
                else:
                    log.warning(
                        "%s: attempt %d failed, retrying: %s",
                        self.name, message.attempts, exc,
                    )
                    self.nack(message)
            else:
                log.info(
                    "%s: done %s in %.1fs",
                    self.name, message.label, time.monotonic() - started,
                )
                stats.acked += 1
                self.ack(message)

            if should_stop is not None and should_stop():
                stats.stopped = True
                break

        return stats

    def drain(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield every waiting message, acking each as it is yielded (no retry safety)."""
        seen = 0
        while limit is None or seen < limit:
            message = self.receive()
            if message is None:
                return
            self.ack(message)
            seen += 1
            yield message.body

    def __enter__(self) -> "Queue":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
