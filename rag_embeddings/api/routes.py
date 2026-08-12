"""
The one route. Everything it needs was built at startup; this is the part that
runs per request.

Declared `def` rather than `async def` on purpose: both halves of a search
block — the embedding forward pass in torch and the query in psycopg — and
Starlette runs a sync endpoint in a threadpool, which keeps a slow query from
stalling the event loop for everyone else.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ..steps.search import search
from .schemas import SearchRequest, SearchResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over the indexed chunks",
)
def post_search(body: SearchRequest, request: Request) -> SearchResponse:
    res = request.app.state.resources

    try:
        # Borrowed for the query and returned immediately after: holding it
        # across the embedding would size the pool by model latency instead of
        # by database work.
        with res.pool.connection() as conn:
            hits = search(
                conn,
                body.query,
                res.embedder,
                limit=body.limit,
                window=body.window,
            )
    except Exception:
        # The failure mode worth distinguishing: the corpus is unreachable, as
        # opposed to a query that legitimately matched nothing. Logged whole,
        # answered with a status rather than the DSN in a traceback.
        log.exception("search failed")
        raise HTTPException(status_code=503, detail="search backend unavailable")

    return SearchResponse(
        query=body.query,
        profile=res.embedder.profile.version,
        count=len(hits),
        hits=hits,
    )
