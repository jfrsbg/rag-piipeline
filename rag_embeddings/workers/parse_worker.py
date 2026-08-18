"""
Step 1 as a job: parse the one document you were given, cache it, announce it
on `to-index`, exit.

    python -m rag_embeddings.workers.parse_worker --uri inbox/a.pdf

That is the whole life of a parse container, and the dispatcher is what starts
it — the arguments above are what `dispatcher.task_argv` writes from the
message it claimed. The same document can arrive as RAG_PARSE_REQUEST in the
environment instead, for an image whose entrypoint is a shell script rather
than this module; either way it is the body of the message a producer
published, unchanged.

It does not read a queue. That is the point of the shape: the dispatcher has
already claimed the message on this container's behalf, so a worker that also
consumed would be a second, uncoordinated consumer of `to-parse` — two
processes racing for the same document, one of them holding a claim nobody will
ack. The only queue opened here is `to-index`, and only to publish onto it.

It does not retry, either. The exit code is the whole report: zero and the
dispatcher acks the message, non-zero and it nacks, and the queue's existing
attempt counting and dead-lettering do the rest. One retry rule, in one place.

Still no database and still no embedding model — this scales on CPU, and how
many run at once is the dispatcher's `--max-in-flight` rather than a replica
count in a deployment.

A parsed document is announced on `to-index` *after* the parse is stored, never
before: the downstream worker may be running on another node and will look for
the parse the instant it sees the message.
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
    """The bytes behind a uri, and a local path Docling can open.

    Local paths are the only scheme wired up, and `local_path` is the same
    function the dispatcher asked before deciding what to mount — so a uri that
    gets a path here is a uri whose file was mounted in, at this exact path.

    An `s3://` branch belongs in the None case — download to a temp file,
    return its path — and it is the only place in the worker that would need to
    change: everything downstream already works from bytes plus a staging path,
    and the dispatcher already mounts nothing for a remote uri.
    """
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
    """One document, then done. The dispatched container's whole life.

    `body` is the message a producer published, exactly as it came off the
    queue — the dispatcher passes it through rather than interpreting it, so
    this is the same dict the old service loop was handed by `Queue.consume`.

    Only `to-index` is opened, and only to publish: the message that named this
    document is held by the dispatcher, which acks it when this process exits
    zero and lets the queue redeliver it when it does not. So there is no ack
    here, no attempt counting and no dead-lettering — the retry rule stays in
    one place, and it is the queue's.

    Raising is how a failure is reported. `main` turns it into a non-zero exit,
    which is the only thing the runner above can see. The queue can be injected,
    which is how the tests run a round trip against `memory://`.
    """
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
