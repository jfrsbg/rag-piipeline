"""
The queue seam.

Workers talk to this interface and never to a broker. A backend supplies
transport — put a message somewhere, hand it to exactly one consumer, take it
back if that consumer dies — and everything built on top of that lives here,
once: the receive loop, acking on success, returning a failure to the queue,
counting attempts and giving up on a message that will never succeed.

That split is the whole point. Swapping the fake backend for SQS should not be
an opportunity to get the retry semantics subtly different.

The contract a backend has to honour:

  * a message given out by `receive` is invisible to other consumers until it
    is acked, nacked, or its visibility deadline passes;
  * `ack` removes it permanently;
  * `nack` makes it visible again with `attempts` incremented;
  * a consumer that dies without doing either gets the message redelivered
    once the deadline passes.

Delivery is therefore at-least-once, never exactly-once, which is fine here:
the pipeline is keyed on content hashes and every write is an upsert, so a
redelivered document converges on the rows it already had.
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
    """One delivery. `receipt` is the backend's handle, opaque above it."""

    body: dict[str, Any]
    receipt: str
    attempts: int = 1
    queue: str = ""

    @property
    def redelivered(self) -> bool:
        return self.attempts > 1

    @property
    def label(self) -> str:
        """Something short enough for a log line and specific enough to grep.

        The receipt alone identifies the delivery but says nothing about the
        document, and following one document across two queues is the reason
        anybody reads these lines. Both message types carry a uri; only
        IndexRequest carries a sha, so this prefers the sha and falls back.
        """
        sha = self.body.get("sha256")
        uri = self.body.get("uri")
        parts = [p for p in (sha[:12] if isinstance(sha, str) else None, uri) if p]
        return f"{self.receipt} ({' '.join(parts)})" if parts else self.receipt


@dataclass
class ConsumeStats:
    """What a worker run did, so the caller can log or exit non-zero."""

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
        """Claim one message, or None if the queue is empty right now.

        Non-blocking: the waiting lives in `consume`, so a backend with real
        long-polling can override `poll` without reimplementing the loop.
        """

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
        """Messages waiting. This is the number an autoscaler scales on."""

    def close(self) -> None:
        """Release whatever the backend holds. Default: nothing to release."""

    # ---------------------------------------------------------------- above

    def publish_all(self, bodies: list[Mapping[str, Any]]) -> list[str]:
        return [self.publish(body) for body in bodies]

    def receive_batch(self, max_messages: int = 10) -> list[Message]:
        """Claim up to `max_messages` at once, or fewer if the queue is thin.

        The default is `receive` in a loop, which is the honest implementation
        for a backend without a batch call. A broker that has one — SQS's
        ReceiveMessage takes MaxNumberOfMessages up to 10 — should override
        this: a batch is one round trip instead of ten, and on SQS it is also
        one request instead of ten to be billed for.

        Only the dispatcher uses it. A worker handles one document at a time
        and gains nothing from claiming a second one it cannot start.
        """
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
        """Receive, waiting up to `timeout` for something to arrive.

        `timeout=None` waits forever, which is what a service wants; a finite
        one is how a container drains a backlog and then exits.

        `should_stop` is checked between polls, so a waiting service reacts to
        a shutdown signal within one `interval` rather than at the end of a
        timeout it may never reach.
        """
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
        """Run `handler` over messages until asked to stop.

        `idle_timeout=None` never returns on its own — a service runs until it
        is signalled. A number is how a container drains a backlog and exits,
        which is also what makes this testable without threads.

        `should_stop` is the signalled case, and is checked only between
        messages: a shutdown arriving mid-document lets that document finish
        and be acked, rather than abandoning a claim for the visibility timeout
        to clean up after the next deploy.

        A handler that raises sends the message back for another attempt, and
        the attempt *before* the one that would exceed `max_attempts` is the
        last: the message is dead-lettered instead of retried forever. A
        document that reliably kills the parser is a poison message, and the
        only useful thing to do with it is set it aside and keep the worker
        alive for the rest of the backlog.
        """
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
        """Every waiting message, acked as it is yielded. For tests and REPLs.

        Not for workers: a consumer that dies mid-iteration has already acked,
        so the message is lost. `consume` is the one with the safety.
        """
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
