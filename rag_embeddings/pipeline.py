"""
Library entry points — same cache, three reprocessing triggers.

These are the in-process API. The two containerised steps in
`rag_embeddings.steps` are the same work cut along the parse/store seam so the
halves can run on different machines: `ingest` is `parse` followed by `index`.
"""

from __future__ import annotations

import logging

from .cache import Manifest, load_cached, parse_and_cache, sha256_of
from .config import Settings
from .embedder import Embedder
from .extraction.chunks import build_chunks
from .extraction.tables import extract_tables
from .storage import sql
from .storage.writer import (
    document_exists,
    document_id_for,
    stale_shas,
    write_all,
)

log = logging.getLogger(__name__)


def ingest(
    conn,
    uri: str,
    blob: bytes,
    mime: str,
    emb: Embedder,
    settings: Settings | None = None,
) -> str:
    """Parse + cache + branch A + branch B, in one process."""
    settings = settings or Settings.from_env()
    sha = sha256_of(blob)
    if document_exists(conn, sha):
        log.info("skip, already ingested %s", sha[:12])
        return sha

    sha, doc = parse_and_cache(uri, blob, settings.cache_dir)
    Manifest.now(sha, uri, mime, settings.parser_version).write(settings.cache_dir)

    tables = extract_tables(doc, document_id=0, parser_version=settings.parser_version)
    chunks = build_chunks(doc, document_id=0, emb=emb)
    write_all(conn, sha, uri, mime, settings.parser_version, tables, chunks)
    return sha


def reextract(conn, sha: str, settings: Settings | None = None) -> None:
    """Extraction schema changed. Branch A only — no GPU, no re-parse."""
    settings = settings or Settings.from_env()
    doc = load_cached(sha, settings.cache_dir)
    document_id = document_id_for(conn, sha)
    tables = extract_tables(doc, document_id, settings.parser_version)
    with conn.transaction():
        conn.execute(sql.TABLE_DELETE, (document_id,))
        conn.cursor().executemany(sql.TABLE_INSERT, tables)


def rechunk(
    conn,
    sha: str,
    emb: Embedder,
    settings: Settings | None = None,
) -> None:
    """Embedding model or chunk config changed. Branch B only."""
    settings = settings or Settings.from_env()
    doc = load_cached(sha, settings.cache_dir)
    document_id = document_id_for(conn, sha)
    chunks = build_chunks(doc, document_id, emb)
    with conn.transaction():
        conn.execute(sql.CHUNK_DELETE, (document_id,))
        conn.cursor().executemany(sql.CHUNK_INSERT, chunks)


def stale_documents(conn, emb: Embedder) -> list[str]:
    """Which documents need rechunk() after a profile change."""
    return stale_shas(conn, emb.profile.version)
