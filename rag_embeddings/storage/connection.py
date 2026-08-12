"""Connection factory. register_vector() is what makes list[float] bind to
`vector`, so it has to happen on every connection, not once per process."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector


def connect(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn


def pool(dsn: str, *, min_size: int = 1, max_size: int = 4):
    """The same connection, pooled — what a long-lived process wants.

    The index service opens one connection and holds it for the life of the
    container, because it handles one message at a time; the API handles
    requests concurrently and must not share one across them, so it borrows and
    returns instead. `configure` is the pool's version of the rule
    above: it runs on each connection the pool opens, including the ones it
    reopens after the database restarts under it.

    Returned unopened. The caller decides when connecting is allowed to fail,
    which for the API is startup rather than the first request.
    """
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
        configure=register_vector,
        open=False,
    )
