"""
`handler` runs the dispatcher loop over the messages in one SQS event.
Requires `ReportBatchItemFailures` on the event source mapping, or one bad
document redelivers the whole batch. `RAG_ACK_ON` defaults to `launch` here.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import replace
from typing import Any, Mapping

from ..config import DispatchSettings, Settings
from ..queues.base import Message, Queue
from ..runners import open_runner
from .dispatcher import Dispatcher, DispatchStats, spec_builder

log = logging.getLogger(__name__)
logging.getLogger().setLevel(os.environ.get("RAG_LOG_LEVEL", "INFO"))

# Seconds kept back from the function's own timeout so that the handler returns
# — reporting whatever it has — instead of being killed with the batch
# unreported. A killed invocation redelivers every message in the batch.
DEADLINE_MARGIN = 15.0


class EventBatch(Queue):
    """The messages in one Lambda event, behind the Queue interface."""

    name = "sqs-event"

    def __init__(self, records: list[Mapping[str, Any]]):
        self._messages = deque(_message(record) for record in records)
        self._ids = [message.receipt for message in self._messages]
        self._acked: set[str] = set()

    def receive(self) -> Message | None:
        return self._messages.popleft() if self._messages else None

    def ack(self, message: Message) -> None:
        self._acked.add(message.receipt)

    def nack(self, message: Message) -> None:
        # Nothing to put back: SQS still holds it, and reporting it as a
        # failure is what makes it visible again after the visibility timeout.
        log.warning("reporting %s for redelivery", message.receipt)

    def dead_letter(self, message: Message, reason: str) -> None:
        # Also just a failure. The queue's own redrive policy owns the DLQ here
        # — parking a message from inside a Lambda would mean deleting it from
        # SQS and writing it somewhere else, which is exactly what redrive
        # already does and does better.
        log.error("out of attempts, leaving to redrive: %s (%s)",
                  message.receipt, reason)

    def depth(self) -> int:
        return len(self._messages)

    def publish(self, body: Mapping[str, Any]) -> str:
        raise NotImplementedError("an event batch is read-only")

    @property
    def failures(self) -> list[dict[str, str]]:
        """The partial batch response: every message not explicitly acked."""
        return [
            {"itemIdentifier": message_id}
            for message_id in self._ids
            if message_id not in self._acked
        ]


def _message(record: Mapping[str, Any]) -> Message:
    """Turn one SQS record into a Message."""
    raw = record.get("body", "")
    if isinstance(raw, Mapping):
        body = dict(raw)                      # already decoded (a test, a fake)
    else:
        try:
            body = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            log.error("message %s is not JSON: %s", record.get("messageId"), exc)
            body = {"_invalid": str(raw)[:200]}
    if not isinstance(body, dict):
        body = {"_invalid": str(body)[:200]}

    attributes = record.get("attributes") or {}
    return Message(
        body=body,
        receipt=str(record.get("messageId", "")),
        attempts=int(attributes.get("ApproximateReceiveCount", 1) or 1),
        queue="sqs-event",
    )


def handler(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """Dispatch one SQS batch and return the partial batch response."""
    records = event.get("Records")
    if records is None:
        log.info("no Records: treating the event as one message body")
        records = [{"messageId": "direct-invoke", "body": event}]

    settings = Settings.from_env()
    dispatch = DispatchSettings.from_env(
        # Lambda is billed for waiting, so the default flips here. See the
        # module docstring.
        ack_on=os.environ.get("RAG_ACK_ON") or "launch",
    )
    dispatch = _fit_to_deadline(dispatch, context)

    queue = EventBatch(list(records))
    runner = open_runner(dispatch.runner_url)
    dispatcher = Dispatcher(
        queue,
        runner,
        spec_builder(settings, dispatch),
        max_in_flight=dispatch.max_in_flight,
        batch_size=dispatch.batch_size,
        ack_on=dispatch.ack_on,
        task_timeout=dispatch.task_timeout,
        # There is no waiting to be done between messages: the batch is
        # finite and already in memory.
        poll_interval=0.2,
        status_interval=1.0,
    )

    try:
        stats: DispatchStats = dispatcher.run(idle_timeout=0.0)
    finally:
        runner.close()

    failures = queue.failures
    log.info(
        "dispatched %d document(s) from %d message(s): %d launched, %d reported back",
        stats.documents, stats.received, stats.launched, len(failures),
    )
    return {"batchItemFailures": failures}


def _fit_to_deadline(dispatch: DispatchSettings, context: Any) -> DispatchSettings:
    """Shorten the task timeout so the handler outlives its own tasks."""
    remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
    if not dispatch.waits or remaining_ms is None:
        return dispatch

    budget = max(1.0, remaining_ms() / 1000.0 - DEADLINE_MARGIN)
    if budget >= dispatch.task_timeout:
        return dispatch

    log.info("cutting the task timeout to %.0fs to fit the invocation", budget)
    return replace(dispatch, task_timeout=budget)
