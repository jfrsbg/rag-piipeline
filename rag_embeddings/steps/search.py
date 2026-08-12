"""
Semantic search — the read side of step 2, and the only check that covers the
query path.

Everything the pipeline writes can look correct while retrieval is broken: a
query embedded through a different model, or without its profile's prefix,
does not raise, it just ranks the wrong passages plausibly. So the query
vector is built from the same `Settings.profile` step 2 indexed with, and each
hit's stored `chunk_config` is compared against it — a mismatch is a warning
on the result, not a silent bad ranking.

No cache and no parser here: this reads the database and embeds one query.
Neither the connection nor the model is owned at this level. Both are passed
in, because the caller is now a service — see `rag_embeddings.api`, which
loads the model once at startup rather than once per query.
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


def warn_on_profile_mismatch(conn, emb: Embedder) -> None:
    """The failure this whole module exists to make visible.

    Checked against the corpus rather than the returned hits, because a query
    embedded through the wrong profile can rank every stale chunk below a
    handful of matching ones and never surface in `limit` rows.

    Public because it is a property of the corpus, not of a query: the service
    runs it once at startup instead of on every request.
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
