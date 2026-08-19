"""
Step 1 as a job: parse the one document named by --uri or RAG_PARSE_REQUEST,
cache it, announce it on `to-index`, exit. It never consumes `to-parse` — the
dispatcher holds that claim — and the exit code is the whole failure report.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from ..cache import Manifest, drop_cached, parse_and_cache, sha256_of
from ..cli import (
    add_document_args,
    add_queue_args,
    common_parser,
    configure_logging,
    document_body_from,
    settings_from,
)
from ..config import Settings
from ..queues import IndexRequest, ParseRequest, Queue, local_path, open_queue
from ..steps.parse import guess_mime

log = logging.getLogger(__name__)


def fetch(uri: str) -> tuple[bytes, str]:
    """Return the bytes behind a uri and a local path Docling can open."""
    path = local_path(uri)
    if path is None:
        raise NotImplementedError(
            f"no fetcher for {uri} — only local paths are wired up; add the "
            f"branch here and nothing else in the worker changes"
        )
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
    body: dict[str, Any],
    settings: Settings | None = None,
    *,
    index_queue: Queue | None = None,
) -> str:
    """Parse the one document in `body` and return its sha; raise on failure."""
    settings = settings or Settings.from_env()
    opened = None
    if index_queue is None:
        index_queue = opened = open_queue(settings.queue_url, settings.index_queue)

    log.info("parse job: %s", body.get("uri"))
    try:
        return handle(body, settings, index_queue)
    finally:
        if opened is not None:
            opened.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-parse-worker",
        parents=[common_parser()],
        description=(
            "Step 1. Parse the one document named by --uri (or by "
            "RAG_PARSE_REQUEST) and exit. This is what the dispatcher starts."
        ),
    )
    add_queue_args(parser)
    add_document_args(parser)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    body = document_body_from(args)
    if body is None:
        # Not a usage nit: a container started with no document would otherwise
        # exit zero having parsed nothing, and the dispatcher would ack the
        # message. Losing a document silently is the one outcome worth being
        # loud about, so this is exit 2 with the reason on stderr.
        parser.error(
            "no document given — pass --uri, or set RAG_PARSE_REQUEST. This "
            "worker is a job: the dispatcher hands it one document and it does "
            "not read a queue."
        )

    # No signal handler and no loop: there is one document, and the exit code
    # is the whole report. A SIGTERM mid-parse is the runner's timeout, and the
    # dispatcher redelivers the message it is still holding.
    try:
        sha = run(body, settings_from(args))
    except Exception:                                       # noqa: BLE001
        # Logged rather than raised so the traceback lands in the container's
        # log, where `runner.logs()` will put it next to the failing document.
        log.exception("failed to parse %s", body.get("uri"))
        return 1

    log.info("parse job done: %s -> %s", body.get("uri"), sha[:12])
    return 0


# See enqueue.py: runnable as a module, so the container needs no install step
# for its entrypoint to resolve. `rag-parse-worker` runs the same main().
if __name__ == "__main__":
    raise SystemExit(main())
