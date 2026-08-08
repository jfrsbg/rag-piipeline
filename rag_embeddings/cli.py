"""
Argument plumbing shared by both steps.

Kept out of the step modules so that each step's file is about what the step
does, and so the flags cannot drift apart between them.
"""

from __future__ import annotations

import argparse
import logging

from .config import Settings


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


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        cache_dir=getattr(args, "cache_dir", None),
        parser_version=getattr(args, "parser_version", None),
        dsn=getattr(args, "dsn", None),
        profile=getattr(args, "profile", None),
        max_tokens=getattr(args, "max_tokens", None),
        headroom=getattr(args, "headroom", None),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
