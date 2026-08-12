"""
The producer, and the coordinator jobs that are not a worker's business.

Two things live here because they are the same thing — deciding what work
exists — and because that decision is centralised on purpose. A worker is
handed one document and never enumerates: `cached_shas` and the staleness query
are whole-cache and whole-table scans, and running either from thirty replicas
at once is how a scan becomes an outage.

    enqueue files    inbox/            -> to-parse    (new documents)
    enqueue cached                     -> to-index    (cache exists, rows do not)
    enqueue stale                      -> to-index    (profile changed, rechunk)

In production `files` is replaced by whatever watches object storage — an S3
event, a Lambda, a cron over a prefix. It publishes the same message either
way, which is why the workers do not care which one you use.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from ..cache import Manifest, cached_shas
from ..cli import add_db_args, add_queue_args, common_parser, configure_logging, settings_from
from ..config import Settings
from ..queues import IndexRequest, ParseRequest, Queue, open_queue
from ..steps.parse import resolve_sources

log = logging.getLogger(__name__)


def enqueue_files(
    sources: Sequence[str],
    settings: Settings | None = None,
    *,
    queue: Queue | None = None,
    pattern: str = "*",
    mime: str | None = None,
    uri_prefix: str | None = None,
    force: bool = False,
) -> list[str]:
    """One `to-parse` message per file. Returns the uris published.

    Reuses step 1's own source resolution so a directory, a glob and an
    explicit path mean here exactly what they mean there.
    """
    settings = settings or Settings.from_env()
    paths = resolve_sources(sources, pattern)
    queue = queue or open_queue(settings.queue_url, settings.parse_queue)

    for path in paths:
        queue.publish(
            ParseRequest(
                uri=str(path), mime=mime, uri_prefix=uri_prefix, force=force
            ).to_body()
        )

    log.info("enqueued %d document(s) on %s", len(paths), settings.parse_queue)
    return [str(p) for p in paths]


def enqueue_cached(
    shas: Sequence[str] | None = None,
    settings: Settings | None = None,
    *,
    queue: Queue | None = None,
    with_tables: bool = True,
    with_chunks: bool = True,
) -> list[str]:
    """One `to-index` message per cached parse.

    The re-entry point after a step 2 change: the parses are already there, so
    this fans them back out without re-reading a single source document.
    """
    settings = settings or Settings.from_env()
    selected = list(shas) if shas else cached_shas(settings.cache_dir)
    queue = queue or open_queue(settings.queue_url, settings.index_queue)

    for sha in selected:
        manifest = Manifest.read(sha, settings.cache_dir)
        queue.publish(
            IndexRequest.from_manifest(
                manifest, with_tables=with_tables, with_chunks=with_chunks
            ).to_body()
        )

    log.info("enqueued %d cached document(s) on %s", len(selected), settings.index_queue)
    return selected


def enqueue_stale(
    settings: Settings | None = None,
    *,
    conn=None,
    queue: Queue | None = None,
) -> list[str]:
    """Documents whose stored chunk_config no longer matches the profile.

    Chunks only: the parse and the tables did not change, so re-running branch
    A would rewrite identical rows and pay for the privilege.
    """
    from ..storage.connection import connect
    from ..storage.writer import stale_shas

    settings = settings or Settings.from_env()
    owns_conn = conn is None
    conn = conn or connect(settings.dsn)
    try:
        shas = stale_shas(conn, settings.profile.version)
    finally:
        if owns_conn:
            conn.close()

    return enqueue_cached(
        shas, settings, queue=queue, with_tables=False, with_chunks=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-enqueue",
        parents=[common_parser()],
        description="Publish work onto a queue. The producer side of the workers.",
    )
    add_queue_args(parser)
    sub = parser.add_subparsers(dest="what", required=True)

    files = sub.add_parser("files", help="new documents -> to-parse")
    files.add_argument("sources", nargs="+", help="files, directories or globs")
    files.add_argument("--pattern", default="*", help="glob applied to directories")
    files.add_argument("--mime", help="override the guessed mime type")
    files.add_argument("--uri-prefix", help="record uris under this prefix")
    files.add_argument("--force", action="store_true", help="drop cached parses first")

    cached = sub.add_parser("cached", help="cached parses -> to-index")
    cached.add_argument("shas", nargs="*", help="default: everything in the cache")
    branches = cached.add_mutually_exclusive_group()
    branches.add_argument("--tables-only", action="store_true")
    branches.add_argument("--chunks-only", action="store_true")

    stale = sub.add_parser("stale", help="documents needing a rechunk -> to-index")
    add_db_args(stale)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    settings = settings_from(args)

    if args.what == "files":
        published = enqueue_files(
            args.sources,
            settings,
            pattern=args.pattern,
            mime=args.mime,
            uri_prefix=args.uri_prefix,
            force=args.force,
        )
    elif args.what == "cached":
        published = enqueue_cached(
            args.shas or None,
            settings,
            with_tables=not args.chunks_only,
            with_chunks=not args.tables_only,
        )
    else:
        published = enqueue_stale(settings)

    for item in published:
        print(item)
    return 0


# `python -m rag_embeddings.workers.enqueue` is the invocation that needs
# nothing installed — only the source tree on sys.path, which is what both the
# container and a checkout have. `rag-enqueue` runs the same main().
if __name__ == "__main__":
    raise SystemExit(main())
