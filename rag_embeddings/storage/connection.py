"""Connection factory. register_vector() is what makes list[float] bind to
`vector`, so it has to happen on every connection, not once per process."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector


def connect(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn
