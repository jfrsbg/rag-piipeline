"""
Step 2 — load the cached parse, extract tables and chunks, store both.

Reads only from the cache and the database: no parser runs here. Tables and
chunks for a document land in a single transaction, so a document is never
left with vectors but no rows.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from ..cache import Manifest, cached_shas, load_cached
from ..cli import (
    add_db_args,
    add_profile_args,
    common_parser,
    configure_logging,
    settings_from,
)
from ..config import Settings
from ..embedder import Embedder
from ..extraction.chunks import build_chunks
from ..extraction.tables import extract_tables
from ..storage.connection import connect
from ..storage.writer import document_exists, stale_shas, write_all

log = logging.getLogger(__name__)


def select_shas(
    conn,
    settings: Settings,
    *,
    shas: Sequence[str] | None,
    stale_only: bool,
    skip_existing: bool,
) -> list[str]:
    """Which cached documents this run should write."""
    selected = list(shas) if shas else cached_shas(settings.cache_dir)

    if stale_only:
        stale = set(stale_shas(conn, settings.profile.version))
        selected = [s for s in selected if s in stale]

    if skip_existing:
        selected = [s for s in selected if not document_exists(conn, s)]

    return selected


def index_document(
    conn,
    sha: str,
    settings: Settings,
    emb: Embedder | None,
    *,
    with_tables: bool = True,
    with_chunks: bool = True,
) -> int:
    """One document: both branches derived from the cache, one commit."""
    doc = load_cached(sha, settings.cache_dir)
    manifest = Manifest.read(sha, settings.cache_dir)

    tables = (
        extract_tables(doc, document_id=0, parser_version=settings.parser_version)
        if with_tables
        else None
    )
    chunks = build_chunks(doc, document_id=0, emb=emb) if with_chunks else None

    return write_all(
        conn,
        sha,
        manifest.uri,
        manifest.mime,
        settings.parser_version,
        tables,
        chunks,
    )


def index_documents(
    shas: Sequence[str] | None = None,
    settings: Settings | None = None,
    *,
    conn=None,
    with_tables: bool = True,
    with_chunks: bool = True,
    stale_only: bool = False,
    skip_existing: bool = False,
) -> list[str]:
    """Extract and store every selected cached document.

    Passing no shas means "everything in the cache". The embedding model is
    loaded once for the whole run, and not at all when chunks are skipped.
    """
    if not (with_tables or with_chunks):
        raise ValueError("nothing to do: both branches disabled")

    settings = settings or Settings.from_env()
    owns_conn = conn is None
    conn = conn or connect(settings.dsn)

    try:
        selected = select_shas(
            conn,
            settings,
            shas=shas,
            stale_only=stale_only,
            skip_existing=skip_existing,
        )
        if not selected:
            log.info("step 2: nothing to index")
            return []

        emb = (
            Embedder(settings.profile, token_budget=settings.embed_token_budget)
            if with_chunks
            else None
        )
        log.info(
            "step 2: %d document(s), branches=%s%s",
            len(selected),
            "tables" if with_tables else "",
            "+chunks" if with_chunks else "",
        )

        written: list[str] = []
        for sha in selected:
            index_document(
                conn, sha, settings, emb,
                with_tables=with_tables, with_chunks=with_chunks,
            )
            written.append(sha)

        log.info("step 2 done: %d document(s) written", len(written))
        return written
    finally:
        if owns_conn:
            conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="step2-index",
        parents=[common_parser()],
        description="Step 2: extract tables and chunks from the cache and store both.",
    )
    add_db_args(parser)
    add_profile_args(parser)

    parser.add_argument(
        "shas",
        nargs="*",
        help="content hashes to index (default: everything in the cache)",
    )
    branches = parser.add_mutually_exclusive_group()
    branches.add_argument(
        "--tables-only",
        action="store_true",
        help="branch A only — extraction schema changed, no model load",
    )
    branches.add_argument(
        "--chunks-only",
        action="store_true",
        help="branch B only — embedding model or chunk config changed",
    )
    parser.add_argument(
        "--stale",
        action="store_true",
        help="restrict to documents whose stored chunk_config differs from the "
             "active profile",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip documents already present in `documents`",
    )

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    written = index_documents(
        args.shas or None,
        settings_from(args),
        with_tables=not args.chunks_only,
        with_chunks=not args.tables_only,
        stale_only=args.stale,
        skip_existing=args.skip_existing,
    )
    for sha in written:
        print(sha)
    return 0
