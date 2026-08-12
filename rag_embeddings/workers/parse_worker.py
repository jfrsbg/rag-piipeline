"""
Step 1 as a service: come up, claim a document, parse it, publish a sha, repeat.

This is the only way documents get parsed. There is no batch driver walking a
directory any more — a container here is handed a single document at a time and
has no idea how many exist, which is exactly what lets N of them run at once
without any of them agreeing on anything.

It is a service rather than a job: it starts with the deployment, blocks on an
empty `to-parse`, and stays up when the backlog drains. Nothing about an empty
queue means the work is finished.

Still no database and still no embedding model — this pool scales on CPU and
the number of pods is decided by the depth of `to-parse`.

A parsed document is announced on `to-index` *after* the parse is stored, never
before: the downstream worker may be running on another node and will look for
the parse the instant it sees the message.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from ..cache import Manifest, drop_cached, parse_and_cache, sha256_of
from ..cli import (
    add_queue_args,
    add_worker_args,
    common_parser,
    configure_logging,
    settings_from,
)
from ..config import Settings
from ..queues import ConsumeStats, IndexRequest, ParseRequest, Queue, open_queue
from ..shutdown import stop_requested
from ..steps.parse import guess_mime

log = logging.getLogger(__name__)


def fetch(uri: str) -> tuple[bytes, str]:
    """The bytes behind a uri, and a local path Docling can open.

    Local paths are the only scheme wired up. An `s3://` branch belongs here —
    download to a temp file, return its path — and it is the only place in the
    worker that would need to change, because everything downstream is already
    working from bytes plus a staging path.
    """
    path = Path(uri)
    if not path.exists():
        raise FileNotFoundError(f"nothing to parse at {uri}")
    return path.read_bytes(), str(path)


def handle(
    body: dict[str, Any],
    settings: Settings,
    index_queue: Queue | None,
) -> str:
    """One message: parse one document, announce it, return its sha."""
    request = ParseRequest.from_body(body)
    blob, source = fetch(request.uri)
    name = Path(request.uri).name

    uri = (
        f"{request.uri_prefix.rstrip('/')}/{name}"
        if request.uri_prefix
        else request.uri
    )

    if request.force:
        drop_cached(sha256_of(blob), settings.cache_dir)

    sha, _doc = parse_and_cache(uri, blob, settings.cache_dir, source=source)
    manifest = Manifest.now(
        sha, uri, guess_mime(name, request.mime), settings.parser_version
    )
    manifest.write(settings.cache_dir)

    if index_queue is not None:
        index_queue.publish(IndexRequest.from_manifest(manifest).to_body())
        log.info("parsed %s -> %s", name, sha[:12])
    return sha


def run(
    settings: Settings | None = None,
    *,
    parse_queue: Queue | None = None,
    index_queue: Queue | None = None,
    max_messages: int | None = None,
    idle_timeout: float | None = None,
    max_attempts: int = 3,
    visibility_timeout: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ConsumeStats:
    """Consume `to-parse`. Returns only when told to stop.

    `idle_timeout` is the exception and exists for the tests and for a drain:
    left unset, an empty queue blocks rather than ends, which is what makes
    this a service. Queues can be injected, which is how the tests run a whole
    round trip in one process against `memory://` without touching argv or the
    environment.
    """
    settings = settings or Settings.from_env()
    opened: list[Queue] = []

    if parse_queue is None:
        kwargs = (
            {} if visibility_timeout is None
            else {"visibility_timeout": visibility_timeout}
        )
        parse_queue = open_queue(settings.queue_url, settings.parse_queue, **kwargs)
        opened.append(parse_queue)
    if index_queue is None:
        index_queue = open_queue(settings.queue_url, settings.index_queue)
        opened.append(index_queue)

    log.info(
        "parse worker up: %r -> %r, %d waiting",
        parse_queue, index_queue, parse_queue.depth(),
    )
    try:
        stats = parse_queue.consume(
            lambda body: handle(body, settings, index_queue),
            max_messages=max_messages,
            idle_timeout=idle_timeout,
            max_attempts=max_attempts,
            should_stop=should_stop,
        )
    finally:
        for queue in opened:
            queue.close()

    log.info(
        "parse service %s: %d parsed, %d failed, %d dead-lettered",
        "stopped" if stats.stopped else "drained",
        stats.acked, stats.failed, stats.dead_lettered,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-parse-worker",
        parents=[common_parser()],
        description="Step 1 as a service: parse one document per message, forever.",
    )
    add_queue_args(parser)
    add_worker_args(parser)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    with stop_requested() as should_stop:
        stats = run(
            settings_from(args),
            max_messages=args.max_messages,
            idle_timeout=args.idle_timeout,
            max_attempts=args.max_attempts,
            visibility_timeout=args.visibility_timeout,
            should_stop=should_stop,
        )

    # Being asked to stop is a clean exit however much was done first —
    # `compose stop` and a rolling update must not look like a crash to the
    # restart policy.
    if stats.stopped:
        return 0
    # A dead letter on its own is not a failure either — the message is parked,
    # the service is healthy, and restarting the pod would only re-park it.
    # Draining without handling anything successfully is different: that is a
    # bad mount or a bad config, and it should be loud.
    return 0 if stats.received == 0 or stats.acked else 1
