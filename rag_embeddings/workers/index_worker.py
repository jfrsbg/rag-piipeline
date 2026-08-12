"""
Step 2 as a service: come up, claim a sha, extract, store, repeat.

The reason this is a service and not a container per document is one line of
`Embedder.__init__` — it loads a multi-gigabyte model. Amortised over a pod's
lifetime that is a startup cost; paid per document it is the pipeline. So the
model and the connection are built once, before the loop, and the loop itself
holds nothing.

The same argument decides the pool sizes. Parse workers are cheap and scale
with the backlog; these carry a resident model and a database connection, so
there are fewer of them and the queue between the two is where the difference
in throughput is allowed to accumulate. That backlog is the point, not a
problem to design away.

Concurrency here is bounded by Postgres, not by CPU: every replica holds a
connection, and the HNSW index on `chunks.embedding` is a shared write
structure that stops rewarding parallelism well before the connection limit
does. Scale this pool on latency, and put a pooler in front of it.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable, Sequence

from ..cache import load_cached
from ..cli import (
    add_db_args,
    add_profile_args,
    add_queue_args,
    add_worker_args,
    common_parser,
    configure_logging,
    settings_from,
)
from ..config import Settings
from ..embedder import Embedder
from ..extraction.chunks import build_chunks
from ..extraction.tables import extract_tables
from ..queues import ConsumeStats, IndexRequest, Queue, open_queue
from ..shutdown import stop_requested
from ..storage.connection import connect
from ..storage.writer import write_all

log = logging.getLogger(__name__)


def handle(
    body: dict[str, Any],
    settings: Settings,
    conn,
    emb: Embedder | None,
) -> int:
    """One message: one document, one transaction, one commit.

    Nothing here selects work. Deciding what needs indexing is a whole-cache or
    whole-table scan, so it belongs to the producer (`workers.enqueue`) and runs
    once; a worker is told what to do and does that only. The Embedder is built
    before the loop for the same reason in reverse — per call it would be the
    pipeline's dominant cost.

    The manifest comes off the message rather than off the cache, so this never
    reads the sidecar the parse service wrote.
    """
    request = IndexRequest.from_body(body)
    manifest = request.to_manifest()
    doc = load_cached(request.sha256, settings.cache_dir)

    tables = (
        extract_tables(doc, document_id=0, parser_version=settings.parser_version)
        if request.with_tables
        else None
    )
    chunks = (
        build_chunks(doc, document_id=0, emb=emb) if request.with_chunks else None
    )
    if tables is None and chunks is None:
        raise ValueError(f"{request.sha256[:12]}: both branches disabled")
    if chunks is not None and emb is None:
        raise RuntimeError(
            "message asks for chunks but this worker started with --tables-only"
        )

    return write_all(
        conn,
        request.sha256,
        manifest.uri,
        manifest.mime,
        settings.parser_version,
        tables,
        chunks,
    )


def run(
    settings: Settings | None = None,
    *,
    index_queue: Queue | None = None,
    conn=None,
    emb: Embedder | None = None,
    with_chunks: bool = True,
    max_messages: int | None = None,
    idle_timeout: float | None = None,
    max_attempts: int = 3,
    visibility_timeout: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ConsumeStats:
    """Consume `to-index`. Returns only when told to stop.

    Everything expensive is a parameter with a lazy default, so a test injects
    fakes and a container builds the real thing — same loop either way. As with
    the parse service, an unset `idle_timeout` means an empty queue blocks
    instead of ending the process.
    """
    settings = settings or Settings.from_env()
    owns_conn = conn is None
    opened: list[Queue] = []

    if index_queue is None:
        kwargs = (
            {} if visibility_timeout is None
            else {"visibility_timeout": visibility_timeout}
        )
        index_queue = open_queue(settings.queue_url, settings.index_queue, **kwargs)
        opened.append(index_queue)

    # Both before the loop. This is the entire point of the worker.
    if emb is None and with_chunks:
        log.info("loading %s", settings.profile.model_id)
        emb = Embedder(settings.profile, token_budget=settings.embed_token_budget)
    conn = conn or connect(settings.dsn)

    log.info("index worker up: %r, %d waiting", index_queue, index_queue.depth())
    try:
        stats = index_queue.consume(
            lambda body: handle(body, settings, conn, emb),
            max_messages=max_messages,
            idle_timeout=idle_timeout,
            max_attempts=max_attempts,
            should_stop=should_stop,
        )
    finally:
        if owns_conn:
            conn.close()
        for queue in opened:
            queue.close()

    log.info(
        "index service %s: %d written, %d failed, %d dead-lettered",
        "stopped" if stats.stopped else "drained",
        stats.acked, stats.failed, stats.dead_lettered,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-index-worker",
        parents=[common_parser()],
        description="Step 2 as a queue worker: one cached document per message.",
    )
    add_db_args(parser)
    add_profile_args(parser)
    add_queue_args(parser)
    add_worker_args(parser)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="never load a model; messages asking for chunks will fail",
    )

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    with stop_requested() as should_stop:
        stats = run(
            settings_from(args),
            with_chunks=not args.tables_only,
            max_messages=args.max_messages,
            idle_timeout=args.idle_timeout,
            max_attempts=args.max_attempts,
            visibility_timeout=args.visibility_timeout,
            should_stop=should_stop,
        )

    # As in the parse service: signalled is a clean exit, drained-without-acking
    # is not. See parse_worker.main for the reasoning.
    if stats.stopped:
        return 0
    return 0 if stats.received == 0 or stats.acked else 1
