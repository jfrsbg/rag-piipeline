"""Connection factory. register_vector() must run on every connection for
list[float] to bind to `vector`."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector


def connect(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn


def pool(dsn: str, *, min_size: int = 1, max_size: int = 4):
    """Build a connection pool for a long-lived process.

    Returned unopened: the caller decides when connecting may fail.
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
