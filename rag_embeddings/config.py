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


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    parser_version: str
    dsn: str
    profile: EmbedProfile
    # Only the workers read these; the two batch steps ignore them entirely.
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
