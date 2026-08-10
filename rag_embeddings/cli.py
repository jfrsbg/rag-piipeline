"""
Argument plumbing shared by both steps.

Kept out of the step modules so that each step's file is about what the step
does, and so the flags cannot drift apart between them.
"""

from __future__ import annotations

import argparse
import logging

from .config import Settings
# From `base` rather than the package, so importing the argument plumbing does
# not drag in a backend — or, through the message types, docling.
from .queues.base import DEFAULT_MAX_ATTEMPTS, DEFAULT_VISIBILITY_TIMEOUT


def common_parser() -> argparse.ArgumentParser:
    """A parent parser with the flags both steps understand."""
    parser = argparse.ArgumentParser(add_help=False)

    env = parser.add_argument_group("environment (all default from env vars)")
    env.add_argument("--cache-dir", help="parse cache root [RAG_CACHE_DIR]")
    env.add_argument("--parser-version", help="stamped on rows [RAG_PARSER_VERSION]")
    env.add_argument("--log-level", default="INFO")

    return parser


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    """Only step 2 loads a model, so only step 2 takes these."""
    group = parser.add_argument_group("embedding profile")
    group.add_argument("--profile", help="named profile or HF model id [RAG_EMBED_PROFILE]")
    group.add_argument("--max-tokens", type=int, help="from the model card [RAG_EMBED_MAX_TOKENS]")
    group.add_argument("--headroom", type=int, help="[RAG_EMBED_HEADROOM]")


def add_db_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("database")
    group.add_argument("--dsn", help="postgres connection string [RAG_DSN]")


def add_queue_args(parser: argparse.ArgumentParser) -> None:
    """Only the workers and the producer speak to a queue."""
    group = parser.add_argument_group("queue")
    group.add_argument(
        "--queue-url",
        help="backend: file://<dir> or memory:// [RAG_QUEUE_URL]",
    )
    group.add_argument("--parse-queue", help="step 1's queue [RAG_PARSE_QUEUE]")
    group.add_argument("--index-queue", help="step 2's queue [RAG_INDEX_QUEUE]")


def add_worker_args(parser: argparse.ArgumentParser) -> None:
    """The knobs that decide when a worker container exits."""
    group = parser.add_argument_group("worker loop")
    group.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="seconds to wait on an empty queue before exiting; omit to run "
             "forever, which is what a Deployment wants",
    )
    group.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="handle at most this many, then exit",
    )
    group.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="deliveries before a message is dead-lettered (default: %(default)s)",
    )
    group.add_argument(
        "--visibility-timeout",
        type=float,
        default=DEFAULT_VISIBILITY_TIMEOUT,
        help="seconds before a claim held by a dead worker is reclaimed "
             "(default: %(default)s)",
    )


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        cache_dir=getattr(args, "cache_dir", None),
        parser_version=getattr(args, "parser_version", None),
        dsn=getattr(args, "dsn", None),
        profile=getattr(args, "profile", None),
        max_tokens=getattr(args, "max_tokens", None),
        headroom=getattr(args, "headroom", None),
        queue_url=getattr(args, "queue_url", None),
        parse_queue=getattr(args, "parse_queue", None),
        index_queue=getattr(args, "index_queue", None),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
