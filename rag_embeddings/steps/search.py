"""
Semantic search over the indexed chunks.

The connection and embedder are passed in; this module owns neither.
"""

from __future__ import annotations

import logging

from ..embedder import Embedder
from ..storage import sql

log = logging.getLogger(__name__)


def search(
    conn,
    query: str,
    emb: Embedder,
    *,
    limit: int = 5,
    window: int = 0,
) -> list[dict]:
    """Return the nearest chunks to `query`, closest first."""
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


def warn_on_profile_mismatch(conn, emb: Embedder) -> None:
    """Log a warning if the corpus was indexed under a different profile."""
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
