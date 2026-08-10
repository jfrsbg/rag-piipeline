"""
The parse cache: one JSON per content hash, plus a sidecar manifest.

The manifest exists because the two steps run as separate processes. Step 1
knows the uri and the mime type; step 2 only gets a sha. Rather than making
step 2 re-derive that from the source (which may no longer be reachable), step 1
writes it next to the parse.

Every function here takes a `cache_dir` that is really "a cache location": a
Path, a uri, or a BlobStore. `open_store` normalises it, so the same call works
against a directory on a laptop and against object storage from a worker, and
callers never grew a second argument for the difference. When workers pass
their sha on the queue the manifest also travels in the message — see
`queues.messages.IndexRequest` — and this sidecar is the fallback for the
single-machine path that has no queue at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from .blobstore import BlobStore, LocalBlobStore, open_store

log = logging.getLogger(__name__)

CacheLocation = BlobStore | Path | str


def sha256_of(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def cache_key(sha: str) -> str:
    return f"{sha}.json"


def manifest_key(sha: str) -> str:
    return f"{sha}.meta.json"


def cache_path(sha: str, cache_dir: CacheLocation) -> Path:
    """The on-disk path of a parse. Local backends only.

    Kept because a path is what the local tooling wants to print, delete or
    stat; anything that needs to work against a remote backend goes through the
    store instead.
    """
    return _local(cache_dir).path_for(cache_key(sha))


def manifest_path(sha: str, cache_dir: CacheLocation) -> Path:
    return _local(cache_dir).path_for(manifest_key(sha))


def _local(cache_dir: CacheLocation) -> LocalBlobStore:
    store = open_store(cache_dir)
    if not isinstance(store, LocalBlobStore):
        raise TypeError(f"{store!r} has no filesystem paths; use the store API")
    return store


@dataclass(frozen=True)
class Manifest:
    """What step 2 needs to know about a parse it did not perform."""

    sha256: str
    uri: str
    mime: str
    parser_version: str
    parsed_at: str

    def write(self, cache_dir: CacheLocation) -> str:
        key = manifest_key(self.sha256)
        open_store(cache_dir).write_bytes(
            key, json.dumps(asdict(self), indent=2).encode()
        )
        return key

    @staticmethod
    def read(sha: str, cache_dir: CacheLocation) -> "Manifest":
        store = open_store(cache_dir)
        key = manifest_key(sha)
        if not store.exists(key):
            raise FileNotFoundError(
                f"no manifest for {sha}; re-run step 1 for this document"
            )
        return Manifest(**json.loads(store.read_bytes(key)))

    @staticmethod
    def now(sha: str, uri: str, mime: str, parser_version: str) -> "Manifest":
        return Manifest(
            sha256=sha,
            uri=uri,
            mime=mime,
            parser_version=parser_version,
            parsed_at=datetime.now(timezone.utc).isoformat(),
        )


def parse_and_cache(
    uri: str,
    blob: bytes,
    cache_dir: CacheLocation,
    *,
    source: str | None = None,
) -> tuple[str, DoclingDocument]:
    """Parse once. Write the JSON before deriving anything, so a crash in a
    downstream branch never costs a re-parse on retry.

    `source` is what Docling actually opens — a local path when `uri` names a
    remote object whose bytes were staged to disk. It defaults to `uri`.

    The cache-hit check is what makes a redelivered queue message cheap rather
    than merely correct: two workers handed the same document both write, the
    store's rename decides the winner, and the bytes are identical either way.
    """
    store = open_store(cache_dir)
    sha = sha256_of(blob)
    key = cache_key(sha)

    if store.exists(key):
        log.info("cache hit %s", sha[:12])
        with store.reading(key) as path:
            return sha, DoclingDocument.load_from_json(path)

    doc = DocumentConverter().convert(source or uri).document
    with store.writing(key) as path:
        doc.save_as_json(path)
    log.info("parsed and cached %s (%s)", sha[:12], uri)
    return sha, doc


def load_cached(sha: str, cache_dir: CacheLocation) -> DoclingDocument:
    store = open_store(cache_dir)
    key = cache_key(sha)
    if not store.exists(key):
        raise FileNotFoundError(f"no cached parse for {sha}; re-run step 1")
    with store.reading(key) as path:
        return DoclingDocument.load_from_json(path)


def drop_cached(sha: str, cache_dir: CacheLocation) -> None:
    """Forget a parse and its manifest — a Docling upgrade, an OCR change."""
    store = open_store(cache_dir)
    store.delete(cache_key(sha))
    store.delete(manifest_key(sha))


def cached_shas(cache_dir: CacheLocation) -> list[str]:
    """Every sha with both a parse and a manifest, in a stable order.

    A whole-cache scan, which is a coordinator's job: a worker is told which
    sha to handle and never enumerates.
    """
    store = open_store(cache_dir)
    manifests = {k[: -len(".meta.json")] for k in store.keys(".meta.json")}
    return sorted(
        key[: -len(".json")]
        for key in store.keys(".json")
        if not key.endswith(".meta.json") and key[: -len(".json")] in manifests
    )
