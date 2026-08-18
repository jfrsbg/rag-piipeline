"""
The dispatcher: one message in, one container per document out.

    queue -> dispatcher -> N parser containers (docker | ECS | k8s)

It is the piece that lets step 1 scale with the backlog instead of with the
deployment. A pool of parse workers is N containers whether the queue holds
zero documents or nine hundred; a dispatcher is one small process that turns
each document into a container and lets the orchestrator decide where it runs.

It owns exactly two things — how a message becomes documents, and how many
containers may exist at once — and delegates the rest:

  * the queue owns retries, attempt counting and dead-lettering (`queues/base`);
  * the runner owns placement and exit codes (`runners/base`);
  * the container owns parsing, which is why the dispatcher imports nothing
    from the pipeline and has no idea what docling is.

A message may hold one document or a thousand. An S3 event notification
arrives as a `Records` array, and a producer batching a directory has every
reason to publish one message with a list rather than a thousand messages. So
unpacking is a fan-out step, and the documents it produces are launched
`max_in_flight` at a time rather than all at once — the alternative is one
message turning into a thousand simultaneous container starts, which is a
denial of service you inflicted on yourself.

The message is acked once every document in it is accounted for. That is what
keeps the guarantees the queue was built with: a dispatcher that dies holding a
claim loses nothing, because the visibility timeout puts the message back and
the whole batch is dispatched again. Re-dispatching is safe — the pipeline is
keyed on content hashes and every write is an upsert — but it is not free, so a
message carrying a large batch wants a visibility timeout longer than its
documents take to parse.

Two ways to run it:

    rag-dispatcher                                     # a service, polling
    rag_embeddings.workers.lambda_dispatch.handler     # an SQS-triggered Lambda

Both run this loop. The Lambda entrypoint differs only in where the messages
come from and in how a failure is reported back — see that module.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote_plus

from ..cli import (
    add_dispatch_args,
    add_queue_args,
    add_worker_args,
    common_parser,
    configure_logging,
    dispatch_settings_from,
    settings_from,
)
from ..config import DispatchSettings, Settings
from ..queues import (
    DEFAULT_MAX_ATTEMPTS,
    Message,
    ParseRequest,
    Queue,
    local_path,
    open_queue,
)
from ..queues.base import DEFAULT_POLL_INTERVAL
from ..runners import (
    DEFAULT_STATUS_INTERVAL,
    Runner,
    TaskHandle,
    TaskResult,
    TaskSpec,
    TaskState,
    open_runner,
    task_name,
)
from ..shutdown import stop_requested

log = logging.getLogger(__name__)


# --------------------------------------------------------------- unpacking

def documents_in(body: Mapping[str, Any]) -> list[ParseRequest]:
    """Every document one message asks for.

    Four shapes, because four things publish onto this queue and only one of
    them is ours:

        {"uri": "..."}                     one document (ParseRequest)
        {"documents": [{...}, {...}]}      a batch the producer grouped
        {"uris": ["...", "..."], ...}      the same, shorthand
        {"Records": [{"s3": {...}}, ...]}  an S3 event notification

    The S3 shape is here rather than in `queues/messages.py` because it is not
    our message: it is what AWS puts on the queue when a bucket notification is
    wired straight to it, and the dispatcher is the only thing that should have
    to know that. Everything downstream sees a ParseRequest either way.

    Anything unreadable raises, which the caller turns into a failed message.
    """
    if "Records" in body:
        return _s3_documents(body["Records"])

    if "documents" in body:
        return [ParseRequest.from_body(item) for item in body["documents"]]

    if "uris" in body:
        # The batch shorthand: one set of options, many uris.
        shared = {k: v for k, v in body.items() if k != "uris"}
        return [ParseRequest.from_body({**shared, "uri": uri}) for uri in body["uris"]]

    return [ParseRequest.from_body(body)]


def _s3_documents(records: Iterable[Mapping[str, Any]]) -> list[ParseRequest]:
    """S3 event records -> `s3://bucket/key` documents.

    Deletions are skipped rather than failed: a bucket notification configured
    for `s3:*` will deliver them, and there is nothing to parse. So is the test
    event S3 sends when the notification is first configured — failing on it
    would dead-letter a message on the day the wiring is set up.
    """
    documents = []
    for record in records:
        event = str(record.get("eventName", ""))
        if event.startswith("ObjectRemoved") or record.get("Event") == "s3:TestEvent":
            log.info("skipping %s", event or "s3:TestEvent")
            continue
        s3 = record.get("s3")
        if not s3:
            raise KeyError("record has no 's3' key; is this an S3 notification?")
        bucket = s3["bucket"]["name"]
        # Keys arrive url-encoded with spaces as '+', which is why this is
        # unquote_plus and not unquote: "Q3 report.pdf" is a real filename and
        # "Q3+report.pdf" is not the object it names.
        key = unquote_plus(s3["object"]["key"])
        documents.append(ParseRequest(uri=f"s3://{bucket}/{key}"))
    return documents


# ------------------------------------------------------------ task building

# What a dispatched container runs. Arguments to `python`, not a command line:
# the image's ENTRYPOINT is the interpreter, so this is what Docker appends
# after it, what Kubernetes sends as `args` and what ECS sends as the command
# override — one string in three places, which is the point of not naming the
# interpreter here.
PARSE_MODULE = "rag_embeddings.workers.parse_worker"


def task_argv(request: ParseRequest) -> tuple[str, ...]:
    """One document -> the arguments the container is started with.

    The message the dispatcher consumed, spelled as flags. `parse_worker` takes
    exactly these and parses that one document; see `cli.add_document_args`.

    The same fields also ride in the environment (`spec_builder` below), and
    both are sent on purpose: this is the readable half — the document a
    container is parsing is in `docker ps` and in `kubectl get job -o yaml`,
    not only in an env var someone has to go and inspect.
    """
    argv = ["-m", PARSE_MODULE, "--uri", request.uri]
    if request.mime:
        argv += ["--mime", request.mime]
    if request.uri_prefix:
        argv += ["--uri-prefix", request.uri_prefix]
    if request.force:
        argv.append("--force")
    return tuple(argv)


def document_mounts(request: ParseRequest) -> tuple[str, ...]:
    """The one document a task may see, as `TaskSpec.mounts`.

    A message names a document by uri and the container has to be able to open
    that uri. For a local one that means a bind mount, and it is the file
    itself rather than the directory holding it: a task is given one document
    and mounting its inbox would hand it every other document as well.

    Mounted at the same absolute path it has on this machine, which is what
    makes the message enough on its own — no translation table between host
    paths and container paths, and the uri stored in the manifest is the uri
    the operator enqueued.

    Nothing is mounted, and that is not a failure, when:

      * the uri is remote (`s3://...`) — the worker fetches it itself, and this
        is the whole of what changes here when that lands;
      * the file is not on this machine — `docker run -v` would silently create
        an empty directory at that path, which is a worse failure than the
        clear one the worker already raises;
      * the uri is relative — there is no absolute path to mount it at, and
        the container's working directory is not the dispatcher's.

    The last two are the operator's mistake rather than the document's, so they
    are logged here: without the line, the only symptom is a container failing
    with "nothing to parse" at a path that plainly exists.
    """
    path = local_path(request.uri)
    if path is None:
        return ()                       # remote: the worker will fetch it

    if not path.is_absolute():
        log.warning(
            "%s is a relative path; a container cannot be given one — enqueue "
            "absolute paths, or an s3:// uri", request.uri,
        )
        return ()
    if not path.exists():
        log.warning("%s is not on this machine; dispatching it anyway", request.uri)
        return ()
    return (f"{path}:{path}:ro",)


def spec_builder(
    settings: Settings, dispatch: DispatchSettings
) -> Callable[[ParseRequest], TaskSpec]:
    """Make the function that turns one document into one container spec.

    The container is told which document it owns three ways, and all three are
    read by the worker at the other end:

        the command                        `-m ...parse_worker --uri ...`
        RAG_PARSE_REQUEST                  the whole message, as JSON
        RAG_DOC_URI / RAG_DOC_MIME / ...   the fields, for a shell entrypoint

    The command is the one that normally does it, and `task_argv` builds it
    from the message. A configured `--task-command` replaces it outright rather
    than being appended to — an image with its own entrypoint is being told
    "start like this", and it still finds the document in the environment.

    Everything else in the environment is what the parser already reads from
    it — cache directory, parser version, the queue it announces onto — passed
    through unchanged so that a container started here and a container started
    by compose are configured identically.

    Naming the document three ways is no use if the container cannot reach it,
    so the spec also carries the mount that puts it there — `document_mounts`
    above, and the reason the operator does not configure an inbox volume.
    """

    def build(request: ParseRequest) -> TaskSpec:
        env = {
            # What the parser reads today. `Settings.from_env` in the container
            # will pick these up exactly as compose's `environment:` block.
            "RAG_CACHE_DIR": str(settings.cache_dir),
            "RAG_PARSER_VERSION": settings.parser_version,
            "RAG_QUEUE_URL": settings.queue_url,
            "RAG_INDEX_QUEUE": settings.index_queue,
            "RAG_PARSE_QUEUE": settings.parse_queue,
            # The document. The container must not read the queue itself — the
            # dispatcher has already claimed this message on its behalf, and a
            # container that also consumed would be a second, uncoordinated
            # consumer of the same queue.
            "RAG_DOC_URI": request.uri,
            "RAG_PARSE_REQUEST": json.dumps(request.to_body()),
            **dispatch.task_env,
        }
        if request.mime:
            env["RAG_DOC_MIME"] = request.mime
        if request.uri_prefix:
            env["RAG_DOC_URI_PREFIX"] = request.uri_prefix
        if request.force:
            env["RAG_DOC_FORCE"] = "1"

        return TaskSpec(
            name=task_name(request.uri),
            image=dispatch.image,
            command=dispatch.task_command or task_argv(request),
            env=env,
            cpu=dispatch.cpu,
            memory_mb=dispatch.memory_mb,
            labels={"rag.doc": request.uri[-120:]},
            mounts=document_mounts(request),
        )

    return build


# ------------------------------------------------------------------- stats

@dataclass
class DispatchStats:
    """What a dispatcher run did, so the caller can log or exit non-zero."""

    received: int = 0          # messages claimed
    documents: int = 0         # documents those messages expanded into
    launched: int = 0          # containers started
    succeeded: int = 0         # containers that exited zero
    failed: int = 0            # containers that did not, plus launches refused
    acked: int = 0             # messages settled and removed
    retried: int = 0           # messages put back for another attempt
    dead_lettered: int = 0     # messages out of attempts
    errors: list[str] = field(default_factory=list)
    # As in ConsumeStats: True when the loop ended because it was signalled
    # rather than because the queue drained.
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.dead_lettered == 0


@dataclass
class _Batch:
    """One claimed message and every task it expanded into.

    The message cannot be acked until all of them are accounted for, so this is
    what "accounted for" is counted in.
    """

    message: Message
    documents: int = 0
    queued: int = 0            # unpacked, not yet launched
    running: int = 0
    failures: list[str] = field(default_factory=list)
    settled: bool = False


@dataclass
class _Task:
    batch: _Batch
    handle: TaskHandle
    started: float
    polled: float = 0.0


class Dispatcher:
    """The loop. Takes a queue and a runner and owns neither."""

    def __init__(
        self,
        queue: Queue,
        runner: Runner,
        spec_for: Callable[[ParseRequest], TaskSpec],
        *,
        max_in_flight: int = 4,
        batch_size: int = 10,
        ack_on: str = "exit",
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        task_timeout: float | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        status_interval: float = DEFAULT_STATUS_INTERVAL,
    ):
        self.queue = queue
        self.runner = runner
        self.spec_for = spec_for
        self.max_in_flight = max(1, max_in_flight)
        self.batch_size = max(1, batch_size)
        self.ack_on = ack_on
        self.max_attempts = max_attempts
        self.task_timeout = task_timeout
        self.poll_interval = poll_interval
        self.status_interval = status_interval

        self._pending: deque[tuple[_Batch, ParseRequest]] = deque()
        self._running: dict[str, _Task] = {}

    # ------------------------------------------------------------- the loop

    def run(
        self,
        *,
        max_messages: int | None = None,
        idle_timeout: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> DispatchStats:
        """Dispatch until told to stop.

        `idle_timeout=None` never returns on its own — a service outlives its
        backlog, exactly as the workers do. A number drains and exits, which is
        what a Lambda invocation and the tests want.

        A stop signal stops *claiming* work immediately and then waits for the
        containers already started, so their messages can be acked rather than
        abandoned for the visibility timeout. A second signal is not caught —
        see `shutdown.py`.
        """
        stats = DispatchStats()
        idle_since = time.monotonic()

        while True:
            stopping = should_stop is not None and should_stop()

            progressed = self._reap(stats)
            if not stopping:
                progressed |= self._launch(stats)
                if not self._pending and self._free() > 0:
                    progressed |= self._receive(stats, max_messages) > 0
                    progressed |= self._launch(stats)

            if progressed:
                idle_since = time.monotonic()
                continue

            # Nothing moved. With ack_on=exit the loop may only end with
            # nothing in flight — anything else abandons a claimed message.
            # With ack_on=launch there is no claim left to abandon: the
            # messages are already acked and the containers were explicitly
            # disowned, so waiting for them would only bill a Lambda for a
            # parse it deliberately did not wait for.
            settled = not self._running or self.ack_on == "launch"
            if settled and not self._pending:
                if stopping:
                    stats.stopped = True
                    break
                if max_messages is not None and stats.received >= max_messages:
                    break
                if idle_timeout is not None and time.monotonic() - idle_since >= idle_timeout:
                    break

            time.sleep(self.poll_interval)

        return stats

    def _free(self) -> int:
        return self.max_in_flight - len(self._running)

    # -------------------------------------------------------------- receive

    def _receive(self, stats: DispatchStats, max_messages: int | None) -> int:
        """Claim messages and unpack them, up to what there is room to run.

        Only called with `_pending` empty. Claiming a message the dispatcher
        has no slot for would start its visibility timer while it sits in a
        deque, and a queue is a much better place to wait than a deque is.
        """
        want = min(self.batch_size, self._free())
        if max_messages is not None:
            want = min(want, max_messages - stats.received)
        if want <= 0:
            return 0

        messages = self.queue.receive_batch(want)
        for message in messages:
            stats.received += 1
            if message.redelivered:
                log.info(
                    "%s: redelivery %d of %s",
                    self.queue.name, message.attempts, message.receipt,
                )
            batch = _Batch(message)
            try:
                requests = documents_in(message.body)
            except Exception as exc:                        # noqa: BLE001
                # A message we cannot read is poison, but it goes through the
                # same attempt rule as everything else rather than being
                # dead-lettered on the spot: one rule, in one place.
                log.error("%s: unreadable message %s: %s",
                          self.queue.name, message.label, exc)
                batch.failures.append(f"unreadable message: {exc}")
                self._settle(batch, stats)
                continue

            if not requests:
                # An S3 test event, or a batch of deletions. Nothing to do, and
                # nothing wrong either — acking is the only sane outcome.
                log.info("%s: %s asked for no documents", self.queue.name, message.label)
                self._settle(batch, stats)
                continue

            batch.documents = batch.queued = len(requests)
            stats.documents += len(requests)
            if len(requests) > 1:
                log.info(
                    "%s: %s carries %d documents",
                    self.queue.name, message.label, len(requests),
                )
            self._pending.extend((batch, request) for request in requests)

        return len(messages)

    # --------------------------------------------------------------- launch

    def _launch(self, stats: DispatchStats) -> bool:
        """Start what fits. Returns whether anything moved."""
        moved = False
        while self._pending and self._free() > 0:
            batch, request = self._pending.popleft()
            batch.queued -= 1
            moved = True
            spec = self.spec_for(request)
            try:
                handle = self.runner.launch(spec)
            except Exception as exc:                        # noqa: BLE001
                # The task never started. That is a failure of the message, not
                # of the document — no capacity, a bad image reference, an
                # expired credential — and every one of those is worth another
                # attempt later.
                log.error("dispatch refused for %s: %s", request.uri, exc)
                stats.failed += 1
                stats.errors.append(f"{request.uri}: {exc}")
                batch.failures.append(f"launch {request.uri}: {exc}")
                self._settle(batch, stats)
                continue

            batch.running += 1
            stats.launched += 1
            now = time.monotonic()
            self._running[handle.id] = _Task(batch=batch, handle=handle, started=now)
            log.info("dispatched %s -> %s", request.uri, handle.label)
            # With ack_on=launch the message is done the moment its last
            # document is accepted; the tasks stay tracked for capacity and for
            # their exit codes, but nothing is waiting on them any more.
            self._settle(batch, stats)
        return moved

    # ----------------------------------------------------------------- reap

    def _reap(self, stats: DispatchStats) -> bool:
        """Ask running tasks how they are doing; settle the ones that finished."""
        moved = False
        now = time.monotonic()
        for task_id, task in list(self._running.items()):
            if now - task.polled < self.status_interval:
                continue
            task.polled = now

            result = self.runner.status(task.handle)
            elapsed = now - task.started
            if not result.state.done:
                # `is not None`, not truthiness: a zero timeout is a legitimate
                # "kill it on the first poll" and only a test ever wants it,
                # but silently ignoring it would hide the timeout path.
                if self.task_timeout is not None and elapsed >= self.task_timeout:
                    log.error("%s overran %.0fs, killing it",
                              task.handle.label, self.task_timeout)
                    self.runner.cancel(task.handle)
                    result = TaskResult(
                        task.handle, TaskState.FAILED,
                        reason=f"timed out after {elapsed:.0f}s",
                    )
                else:
                    continue

            del self._running[task_id]
            task.batch.running -= 1
            moved = True

            if result.ok:
                stats.succeeded += 1
                log.info("%s finished in %.1fs", task.handle.label, elapsed)
            else:
                stats.failed += 1
                stats.errors.append(f"{task.handle.name}: {result.describe()}")
                task.batch.failures.append(f"{task.handle.name}: {result.describe()}")
                log.error("%s failed after %.1fs: %s",
                          task.handle.label, elapsed, result.describe())
                # The container's own last words, next to the document that
                # produced them — otherwise this line is the only record and it
                # says nothing about why.
                tail = self.runner.logs(task.handle)
                if tail:
                    log.error("%s said:\n%s", task.handle.label, tail)

            self.runner.cleanup(task.handle)
            self._settle(task.batch, stats)
        return moved

    # --------------------------------------------------------------- settle

    def _settle(self, batch: _Batch, stats: DispatchStats) -> None:
        """Ack, retry or dead-letter a message once its documents are done.

        The attempt rule is `Queue.consume`'s, deliberately to the letter: the
        attempt before the one that would exceed `max_attempts` is the last,
        and a message that keeps failing is parked rather than retried forever.
        Two implementations of this rule that disagree would be a way for a
        document to be attempted twice as often on one path as the other.
        """
        if batch.settled or batch.queued:
            return
        # Waiting for exits is what makes the ack meaningful; with ack_on=launch
        # there is nothing left to wait for.
        if self.ack_on == "exit" and batch.running:
            return

        batch.settled = True
        message = batch.message
        if not batch.failures:
            self.queue.ack(message)
            stats.acked += 1
            log.info("%s: done %s (%d document(s))",
                     self.queue.name, message.label, batch.documents)
            return

        reason = "; ".join(batch.failures)
        if message.attempts >= self.max_attempts:
            stats.dead_lettered += 1
            log.error("%s: dead-lettering after %d attempts: %s",
                      self.queue.name, message.attempts, reason)
            self.queue.dead_letter(message, reason)
        else:
            stats.retried += 1
            log.warning("%s: attempt %d failed, retrying: %s",
                        self.queue.name, message.attempts, reason)
            # Every document in the message is retried, including any that
            # succeeded. That is safe — the parse cache is keyed on the content
            # hash, so a re-dispatched document that is already parsed costs a
            # container start and no work — and it is the only option a queue
            # with one receipt per message offers.
            self.queue.nack(message)


# -------------------------------------------------------------- entrypoints

def run(
    settings: Settings | None = None,
    dispatch: DispatchSettings | None = None,
    *,
    queue: Queue | None = None,
    runner: Runner | None = None,
    max_messages: int | None = None,
    idle_timeout: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    visibility_timeout: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> DispatchStats:
    """Dispatch `to-parse` onto containers. Returns only when told to stop.

    Queue and runner can be injected, which is how the tests run a whole
    round trip in one process against `memory://` without touching argv or the
    environment.
    """
    settings = settings or Settings.from_env()
    dispatch = dispatch or DispatchSettings.from_env()

    opened_queue = opened_runner = None
    if queue is None:
        kwargs = (
            {} if visibility_timeout is None
            else {"visibility_timeout": visibility_timeout}
        )
        queue = opened_queue = open_queue(
            settings.queue_url, settings.parse_queue, **kwargs
        )
    if runner is None:
        runner = opened_runner = open_runner(dispatch.runner_url)

    dispatcher = Dispatcher(
        queue,
        runner,
        spec_builder(settings, dispatch),
        max_in_flight=dispatch.max_in_flight,
        batch_size=dispatch.batch_size,
        ack_on=dispatch.ack_on,
        max_attempts=max_attempts,
        task_timeout=dispatch.task_timeout,
    )

    log.info(
        "dispatcher up: %r -> %r, image=%s, %d at a time, ack on %s, %d waiting",
        queue, runner, dispatch.image, dispatch.max_in_flight, dispatch.ack_on,
        queue.depth(),
    )
    try:
        stats = dispatcher.run(
            max_messages=max_messages,
            idle_timeout=idle_timeout,
            should_stop=should_stop,
        )
    finally:
        if opened_runner is not None:
            opened_runner.close()
        if opened_queue is not None:
            opened_queue.close()

    log.info(
        "dispatcher %s: %d message(s), %d document(s), %d launched, "
        "%d succeeded, %d failed, %d retried, %d dead-lettered",
        "stopped" if stats.stopped else "drained",
        stats.received, stats.documents, stats.launched,
        stats.succeeded, stats.failed, stats.retried, stats.dead_lettered,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-dispatcher",
        parents=[common_parser()],
        description=(
            "Read documents off a queue and run one parser container for each."
        ),
    )
    add_queue_args(parser)
    add_worker_args(parser)
    add_dispatch_args(parser)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    with stop_requested() as should_stop:
        stats = run(
            settings_from(args),
            dispatch_settings_from(args),
            max_messages=args.max_messages,
            idle_timeout=args.idle_timeout,
            max_attempts=args.max_attempts,
            visibility_timeout=args.visibility_timeout,
            should_stop=should_stop,
        )

    # The same rule the workers use: being signalled is a clean exit however
    # much was done first, a dead letter is a parked document rather than a
    # broken service, and draining without dispatching anything successfully is
    # a bad config that should be loud.
    if stats.stopped:
        return 0
    return 0 if stats.received == 0 or stats.acked else 1


# As elsewhere: runnable as a module so the container needs no install step for
# its entrypoint to resolve. `rag-dispatcher` runs the same main().
if __name__ == "__main__":
    raise SystemExit(main())
