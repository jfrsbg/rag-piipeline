"""The write side. Both branches land in one commit."""

from __future__ import annotations

import logging

from . import sql

log = logging.getLogger(__name__)


def write_all(
    conn,
    sha: str,
    uri: str,
    mime: str,
    parser_version: str,
    tables: list[dict] | None,
    chunks: list[dict] | None,
) -> int:
    """Both branches in one commit. Otherwise a failure between them leaves a
    document with vectors but no rows, and nothing knows it is half-ingested.

    A branch passed as None is left untouched; passing an empty list still
    clears the existing rows, which is what a document with no tables means.
    """
    with conn.transaction():
        cur = conn.cursor()
        cur.execute(sql.DOC_UPSERT, {
            "sha256": sha, "uri": uri, "mime": mime,
            "parser_version": parser_version,
        })
        document_id = cur.fetchone()[0]

        if tables is not None:
            cur.execute(sql.TABLE_DELETE, (document_id,))
            for r in tables:
                r["document_id"] = document_id
            cur.executemany(sql.TABLE_INSERT, tables)

        if chunks is not None:
            cur.execute(sql.CHUNK_DELETE, (document_id,))
            for r in chunks:
                r["document_id"] = document_id
            cur.executemany(sql.CHUNK_INSERT, chunks)

    log.info(
        "wrote %s: %d tables, %d chunks",
        sha[:12], len(tables or []), len(chunks or []),
    )
    return document_id


def document_id_for(conn, sha: str) -> int:
    row = conn.execute(sql.DOC_ID_BY_SHA, (sha,)).fetchone()
    if row is None:
        raise LookupError(f"unknown document {sha}")
    return row[0]


def document_exists(conn, sha: str) -> bool:
    return conn.execute(sql.DOC_EXISTS, (sha,)).fetchone() is not None


def stale_shas(conn, chunk_config: str) -> list[str]:
    """Documents whose stored chunk_config no longer matches the active one."""
    rows = conn.execute(sql.STALE_BY_CHUNK_CONFIG, (chunk_config,)).fetchall()
    return [r[0] for r in rows]
