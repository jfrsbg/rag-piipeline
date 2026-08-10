"""
Semantic search — the read side of step 2, and the only check that covers the
query path.

Everything the pipeline writes can look correct while retrieval is broken: a
query embedded through a different model, or without its profile's prefix,
does not raise, it just ranks the wrong passages plausibly. So the query
vector is built from the same `Settings.profile` step 2 indexed with, and each
hit's stored `chunk_config` is compared against it — a mismatch is a warning
on the result, not a silent bad ranking.

No cache and no parser here: this reads the database and loads one model.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from typing import Sequence

from ..cli import (
    add_db_args,
    add_profile_args,
    common_parser,
    configure_logging,
    settings_from,
)
from ..config import Settings
from ..embedder import Embedder
from ..storage import sql
from ..storage.connection import connect

log = logging.getLogger(__name__)


def search(
    conn,
    query: str,
    emb: Embedder,
    *,
    limit: int = 5,
    window: int = 0,
) -> list[dict]:
    """Nearest chunks to `query`, closest first.

    `window` widens each hit by that many chunks either side. It is a second
    lookup rather than part of the ranking: chunks are stored without overlap,
    so neighbouring text is context to read, never a reason a hit scored.
    """
    qvec = emb.encode_query(query)

    rows = conn.execute(
        sql.CHUNK_SEARCH, {"qvec": qvec, "limit": limit}
    ).fetchall()

    hits = []
    for (cid, doc_id, ord_, page, text, heading_path,
         chunk_config, uri, dist) in rows:
        hit = {
            "id": cid,
            "document_id": doc_id,
            "ord": ord_,
            "page": page,
            "text": text,
            "heading_path": heading_path or [],
            "uri": uri,
            "chunk_config": chunk_config,
            "distance": float(dist),
            "stale": chunk_config != emb.profile.version,
        }
        if window:
            hit["window"] = [
                {"ord": o, "text": t}
                for o, t in conn.execute(
                    sql.CHUNK_WINDOW,
                    {"document_id": doc_id, "ord": ord_, "window": window},
                ).fetchall()
            ]
        hits.append(hit)
    return hits


def _warn_on_profile_mismatch(conn, emb: Embedder) -> None:
    """The failure this whole module exists to make visible.

    Checked against the corpus rather than the returned hits, because a query
    embedded through the wrong profile can rank every stale chunk below a
    handful of matching ones and never surface in `limit` rows.
    """
    stored = {r[0] for r in conn.execute(sql.CHUNK_CONFIGS).fetchall()}
    if not stored:
        log.warning("no chunks in the database — run step 2 first")
    elif emb.profile.version not in stored:
        log.warning(
            "querying with %s but the corpus was indexed with %s; results will "
            "rank badly rather than fail",
            emb.profile.version, ", ".join(sorted(stored)),
        )
    elif len(stored) > 1:
        log.warning(
            "corpus holds more than one chunk_config (%s); long and short "
            "chunks do not rank coherently against each other",
            ", ".join(sorted(stored)),
        )


def run_search(
    query: str,
    settings: Settings | None = None,
    *,
    conn=None,
    limit: int = 5,
    window: int = 0,
) -> list[dict]:
    settings = settings or Settings.from_env()
    owns_conn = conn is None
    conn = conn or connect(settings.dsn)
    try:
        emb = Embedder(settings.profile, token_budget=settings.embed_token_budget)
        _warn_on_profile_mismatch(conn, emb)
        return search(conn, query, emb, limit=limit, window=window)
    finally:
        if owns_conn:
            conn.close()


def format_hits(hits: list[dict], *, width: int = 88) -> str:
    if not hits:
        return "no hits"

    out = []
    for rank, hit in enumerate(hits, 1):
        where = " > ".join(hit["heading_path"]) or "—"
        page = f"p{hit['page']}" if hit["page"] is not None else "p?"
        out.append(
            f"{rank:>2}. {hit['distance']:.4f}  {hit['uri']}  "
            f"[{page} ord={hit['ord']}]{'  STALE' if hit['stale'] else ''}\n"
            f"    {where}"
        )
        window = hit.get("window")
        body = "\n\n".join(w["text"] for w in window) if window else hit["text"]
        out.append(
            textwrap.indent(
                textwrap.fill(" ".join(body.split()), width=width), "    "
            )
        )
        out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-search",
        parents=[common_parser()],
        description="Semantic search over the indexed chunks.",
    )
    add_db_args(parser)
    add_profile_args(parser)

    parser.add_argument("query", help="the question, in the corpus's language")
    parser.add_argument(
        "-k", "--limit", type=int, default=5, help="hits to return (default: %(default)s)"
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=0,
        help="also print this many chunks either side of each hit",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    hits = run_search(
        args.query,
        settings_from(args),
        limit=args.limit,
        window=args.window,
    )

    if args.json:
        import json

        print(json.dumps(hits, indent=2, ensure_ascii=False))
    else:
        print(format_hits(hits))
    return 0
