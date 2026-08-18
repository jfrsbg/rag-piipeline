"""
Argument plumbing shared by the producer and both services.

Kept out of their modules so that each file is about what that process does,
and so the flags cannot drift apart between them.
"""

from __future__ import annotations

import argparse
import logging
import shlex

from .config import (
    DispatchSettings,
    Settings,
    idle_timeout_from_env,
    parse_request_from_env,
)
# From `base` rather than the package, so importing the argument plumbing does
# not drag in a backend — or, through the message types, docling.
from .queues.base import DEFAULT_MAX_ATTEMPTS, DEFAULT_VISIBILITY_TIMEOUT


def common_parser() -> argparse.ArgumentParser:
    """A parent parser with the flags every entrypoint understands."""
    parser = argparse.ArgumentParser(add_help=False)

    env = parser.add_argument_group("environment (all default from env vars)")
    env.add_argument("--cache-dir", help="parse cache root [RAG_CACHE_DIR]")
    env.add_argument("--parser-version", help="stamped on rows [RAG_PARSER_VERSION]")
    env.add_argument("--log-level", default="INFO")

    return parser


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    """Only the index service loads a model, so only it takes these."""
    group = parser.add_argument_group("embedding profile")
    group.add_argument("--profile", help="named profile or HF model id [RAG_EMBED_PROFILE]")
    group.add_argument("--max-tokens", type=int, help="from the model card [RAG_EMBED_MAX_TOKENS]")
    group.add_argument("--headroom", type=int, help="[RAG_EMBED_HEADROOM]")
    group.add_argument(
        "--embed-token-budget",
        type=int,
        help="tokens per forward pass, padding included; lower this if the "
             "worker is OOM-killed [RAG_EMBED_TOKEN_BUDGET]",
    )


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
    """The knobs that decide when a service container exits."""
    group = parser.add_argument_group("worker loop")
    group.add_argument(
        "--idle-timeout",
        type=float,
        default=idle_timeout_from_env(),
        help="seconds to wait on an empty queue before exiting; leave unset — "
             "a service outlives its backlog. Set it only to drain a queue and "
             "stop [RAG_IDLE_TIMEOUT]",
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


def add_document_args(parser: argparse.ArgumentParser) -> None:
    """One document, named on the command line: the parse worker's job mode.

    These are the arguments `dispatcher.task_argv` writes, and the only reason
    they exist. Given `--uri` the worker parses that document and exits;
    without it, it falls back to consuming `to-parse` as a service.
    """
    group = parser.add_argument_group("one document (job mode)")
    group.add_argument(
        "--uri",
        help="parse this document and exit, instead of consuming a queue "
             "[RAG_DOC_URI / RAG_PARSE_REQUEST]",
    )
    group.add_argument("--mime", help="content type, if the name does not say")
    group.add_argument("--uri-prefix", help="rewrite the stored uri under this prefix")
    group.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="re-parse even if the cache already holds this content",
    )


def document_body_from(args: argparse.Namespace) -> dict | None:
    """The message body the worker was handed, from argv or the environment.

    `None` means it was handed nothing and should run as a service. The flags
    win over the environment, as everywhere else here.
    """
    if getattr(args, "uri", None):
        return {
            "uri": args.uri,
            "mime": getattr(args, "mime", None),
            "uri_prefix": getattr(args, "uri_prefix", None),
            "force": bool(getattr(args, "force", None)),
        }
    return parse_request_from_env()


def add_dispatch_args(parser: argparse.ArgumentParser) -> None:
    """Only the dispatcher starts containers, so only it takes these."""
    group = parser.add_argument_group("dispatch")
    group.add_argument(
        "--runner-url",
        help="where tasks run: docker://, process://, ecs://<cluster>/<task-def>, "
             "k8s://<namespace>, memory:// [RAG_RUNNER_URL]",
    )
    group.add_argument("--image", help="parser image [RAG_PARSER_IMAGE]")
    group.add_argument(
        "--task-command",
        help="override the image entrypoint, as one shell-quoted string "
             "[RAG_TASK_COMMAND]",
    )
    group.add_argument(
        "--task-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment for every task; repeatable [RAG_TASK_ENV]",
    )
    group.add_argument("--task-cpu", type=float, help="cores per task [RAG_TASK_CPU]")
    group.add_argument(
        "--task-memory", type=int, help="MiB per task [RAG_TASK_MEMORY_MB]"
    )
    group.add_argument(
        "--max-in-flight",
        type=int,
        help="containers running at once (default: 4) [RAG_MAX_IN_FLIGHT]",
    )
    group.add_argument(
        "--batch-size", type=int, help="messages claimed per receive [RAG_DISPATCH_BATCH]"
    )
    group.add_argument(
        "--ack-on",
        choices=["exit", "launch"],
        help="'exit' holds the message until the container succeeds; 'launch' "
             "acks as soon as it starts [RAG_ACK_ON]",
    )
    group.add_argument(
        "--task-timeout",
        type=float,
        help="seconds before a task is killed and its document retried "
             "[RAG_TASK_TIMEOUT]",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="use memory:// — print what would be launched, launch nothing",
    )


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        cache_dir=getattr(args, "cache_dir", None),
        parser_version=getattr(args, "parser_version", None),
        dsn=getattr(args, "dsn", None),
        profile=getattr(args, "profile", None),
        max_tokens=getattr(args, "max_tokens", None),
        headroom=getattr(args, "headroom", None),
        embed_token_budget=getattr(args, "embed_token_budget", None),
        queue_url=getattr(args, "queue_url", None),
        parse_queue=getattr(args, "parse_queue", None),
        index_queue=getattr(args, "index_queue", None),
    )


def dispatch_settings_from(args: argparse.Namespace) -> DispatchSettings:
    """The dispatch half of the flags. Same rule: `None` means "not passed"."""
    command = getattr(args, "task_command", None)
    return DispatchSettings.from_env(
        runner_url=(
            "memory://" if getattr(args, "dry_run", False)
            else getattr(args, "runner_url", None)
        ),
        image=getattr(args, "image", None),
        task_command=shlex.split(command) if command else None,
        task_env=dict(_env_pairs(getattr(args, "task_env", []) or [])),
        cpu=getattr(args, "task_cpu", None),
        memory_mb=getattr(args, "task_memory", None),
        max_in_flight=getattr(args, "max_in_flight", None),
        batch_size=getattr(args, "batch_size", None),
        ack_on=getattr(args, "ack_on", None),
        task_timeout=getattr(args, "task_timeout", None),
    )


def _env_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--task-env expects KEY=VALUE, got {item!r}")
        pairs.append((key, value))
    return pairs


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
