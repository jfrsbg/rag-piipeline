"""
Runtime configuration.

Everything that differs between a laptop and a container is read from the
environment here, once, so no module below reaches for os.environ on its own.
CLI flags override the environment; the environment overrides the defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .profiles import EmbedProfile, resolve

DEFAULT_CACHE_DIR = "cache/parsed"
# Bump this whenever the pinned docling range in requirements.txt moves: it is
# the stamp that tells you which rows predate a parser upgrade.
DEFAULT_PARSER_VERSION = "docling-2.118"
DEFAULT_PROFILE = "bge-m3"
DEFAULT_DSN = "postgresql://localhost/docs"
DEFAULT_QUEUE_URL = "file://./queue"
DEFAULT_PARSE_QUEUE = "to-parse"
DEFAULT_INDEX_QUEUE = "to-index"
# Tokens per forward pass, padding included — see rag_embeddings/embedder.py
# for what the number buys. It is hardware sizing, not model semantics, which
# is why it lives here and not on the profile: changing it must not change the
# vectors, and must not change `profile.version` stamped on the rows.
DEFAULT_EMBED_TOKEN_BUDGET = 16384
# The read side, served rather than run: see rag_embeddings/api. 0.0.0.0
# because the only caller is outside the container.
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_API_POOL_SIZE = 4


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


def idle_timeout_from_env() -> float | None:
    """Seconds a worker waits on an empty queue before exiting; None is forever.

    Not on `Settings`: an exit condition is per-container rather than
    per-deployment — two replicas of the same service may legitimately disagree
    about it, one draining while the other stays up. It is read here anyway so
    that the rule about os.environ living in one module holds.

    Unset and empty both mean forever, which is the normal case now that the
    services are the only way work is processed, and also what
    `${RAG_IDLE_TIMEOUT:-}` in compose expands to.
    """
    raw = os.environ.get("RAG_IDLE_TIMEOUT")
    return float(raw) if raw not in (None, "") else None


@dataclass(frozen=True)
class ApiSettings:
    """How the service is served, as opposed to what it serves.

    Separate from `Settings` because none of it reaches the search path: the
    same query against the same corpus returns the same hits whatever port
    answered. It is here rather than in the API package only to keep the rule
    that os.environ is read in one module.
    """

    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    log_level: str = "INFO"
    # Requests are served from a threadpool, so connections must be borrowed
    # per request rather than shared. Past a handful the embedding forward
    # pass, not the database, is the limit.
    pool_size: int = DEFAULT_API_POOL_SIZE

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            host=os.environ.get("RAG_API_HOST") or DEFAULT_API_HOST,
            port=_int_env("RAG_API_PORT") or DEFAULT_API_PORT,
            log_level=os.environ.get("RAG_LOG_LEVEL") or "INFO",
            pool_size=_int_env("RAG_API_POOL_SIZE") or DEFAULT_API_POOL_SIZE,
        )


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    parser_version: str
    dsn: str
    profile: EmbedProfile
    embed_token_budget: int = DEFAULT_EMBED_TOKEN_BUDGET
    # Read by the producer and both services; the API never opens a queue.
    queue_url: str = DEFAULT_QUEUE_URL
    parse_queue: str = DEFAULT_PARSE_QUEUE
    index_queue: str = DEFAULT_INDEX_QUEUE

    @classmethod
    def from_env(
        cls,
        *,
        cache_dir: str | None = None,
        parser_version: str | None = None,
        dsn: str | None = None,
        profile: str | None = None,
        max_tokens: int | None = None,
        headroom: int | None = None,
        embed_token_budget: int | None = None,
        queue_url: str | None = None,
        parse_queue: str | None = None,
        index_queue: str | None = None,
    ) -> "Settings":
        """Build settings from the environment, with optional overrides.

        Every keyword is the CLI's chance to win; `None` means "not passed".
        """
        return cls(
            cache_dir=Path(
                cache_dir
                or os.environ.get("RAG_CACHE_DIR")
                or DEFAULT_CACHE_DIR
            ),
            parser_version=(
                parser_version
                or os.environ.get("RAG_PARSER_VERSION")
                or DEFAULT_PARSER_VERSION
            ),
            dsn=(
                dsn
                or os.environ.get("RAG_DSN")
                or os.environ.get("DATABASE_URL")
                or DEFAULT_DSN
            ),
            profile=resolve(
                profile or os.environ.get("RAG_EMBED_PROFILE") or DEFAULT_PROFILE,
                max_tokens=(
                    max_tokens
                    if max_tokens is not None
                    else _int_env("RAG_EMBED_MAX_TOKENS")
                ),
                headroom=(
                    headroom
                    if headroom is not None
                    else _int_env("RAG_EMBED_HEADROOM")
                ),
            ),
            embed_token_budget=(
                embed_token_budget
                if embed_token_budget is not None
                else _int_env("RAG_EMBED_TOKEN_BUDGET") or DEFAULT_EMBED_TOKEN_BUDGET
            ),
            queue_url=(
                queue_url or os.environ.get("RAG_QUEUE_URL") or DEFAULT_QUEUE_URL
            ),
            parse_queue=(
                parse_queue
                or os.environ.get("RAG_PARSE_QUEUE")
                or DEFAULT_PARSE_QUEUE
            ),
            index_queue=(
                index_queue
                or os.environ.get("RAG_INDEX_QUEUE")
                or DEFAULT_INDEX_QUEUE
            ),
        )
