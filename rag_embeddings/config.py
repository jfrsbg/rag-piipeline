"""Runtime configuration.

The only module that reads os.environ. CLI flags override the environment;
the environment overrides the defaults.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profiles import EmbedProfile, resolve
# From `runners.base` rather than the package, and only for the constant: the
# defaults belong beside everything else the environment can set, and reading
# them must not depend on a backend being importable.
from .runners.base import DEFAULT_TASK_TIMEOUT

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
# The dispatcher: what it starts, where, and how many at once. Duplicated from
# rag_embeddings/runners/__init__.py the way DEFAULT_QUEUE_URL is duplicated
# from the queues package — this module is the list of what the environment can
# set, and it should read as one.
DEFAULT_RUNNER_URL = "docker://"
DEFAULT_PARSER_IMAGE = "parse-worker:latest"
DEFAULT_MAX_IN_FLIGHT = 4
DEFAULT_BATCH_SIZE = 10


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else None


def idle_timeout_from_env() -> float | None:
    """Seconds a worker waits on an empty queue before exiting; None is forever."""
    raw = os.environ.get("RAG_IDLE_TIMEOUT")
    return float(raw) if raw not in (None, "") else None


def parse_request_from_env() -> dict[str, Any] | None:
    """The one document a dispatched container was started for; None to run as a service."""
    raw = os.environ.get("RAG_PARSE_REQUEST")
    if raw:
        return json.loads(raw)

    uri = os.environ.get("RAG_DOC_URI")
    if not uri:
        return None
    return {
        "uri": uri,
        "mime": os.environ.get("RAG_DOC_MIME") or None,
        "uri_prefix": os.environ.get("RAG_DOC_URI_PREFIX") or None,
        "force": os.environ.get("RAG_DOC_FORCE", "") not in ("", "0"),
    }


@dataclass(frozen=True)
class ApiSettings:
    """How the API is served, as opposed to what it serves."""

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
class DispatchSettings:
    """How the dispatcher turns a message into a container."""

    # Where tasks run. The only place a backend is named — see
    # rag_embeddings/runners/__init__.py.
    runner_url: str = DEFAULT_RUNNER_URL
    # What to run. Ignored by `ecs://` (the image is in the task definition)
    # and by `process://` (there is no image).
    image: str = DEFAULT_PARSER_IMAGE
    # Empty — the normal case — means the dispatcher builds the command from
    # the message: `-m rag_embeddings.workers.parse_worker --uri ...`, appended
    # to the image's `python` entrypoint. See `dispatcher.task_argv`. Set it
    # only for an image that starts differently; that image then finds the
    # document in RAG_PARSE_REQUEST instead of in its arguments.
    task_command: tuple[str, ...] = ()
    task_env: dict[str, str] = field(default_factory=dict)
    cpu: float | None = None
    memory_mb: int | None = None
    # Containers in flight at once. This is the knob that decides whether a
    # 900-document backlog is a fan-out or an outage: every task is a container
    # pulling an image and holding memory, and nothing downstream of here
    # applies any backpressure of its own.
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    # Messages claimed per receive. Ten is SQS's maximum for one ReceiveMessage
    # call and a sensible ceiling for the others.
    batch_size: int = DEFAULT_BATCH_SIZE
    # "exit"   — hold the message until the container succeeds. At-least-once:
    #            a container that dies gets its document redelivered.
    # "launch" — ack as soon as the task is accepted. Cheaper (and the only
    #            sane choice in a Lambda, which is billed for waiting), but a
    #            container that dies takes its document with it.
    ack_on: str = "exit"
    task_timeout: float = DEFAULT_TASK_TIMEOUT

    @property
    def waits(self) -> bool:
        return self.ack_on == "exit"

    @classmethod
    def from_env(
        cls,
        *,
        runner_url: str | None = None,
        image: str | None = None,
        task_command: Sequence[str] | None = None,
        task_env: Mapping[str, str] | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        ack_on: str | None = None,
        task_timeout: float | None = None,
    ) -> "DispatchSettings":
        command = task_command
        if command is None:
            raw = os.environ.get("RAG_TASK_COMMAND")
            command = shlex.split(raw) if raw else ()

        env = dict(_env_pairs(os.environ.get("RAG_TASK_ENV")))
        env.update(task_env or {})

        ack = (ack_on or os.environ.get("RAG_ACK_ON") or "exit").lower()
        if ack not in ("exit", "launch"):
            raise ValueError(f"RAG_ACK_ON must be 'exit' or 'launch', not {ack!r}")

        return cls(
            runner_url=(
                runner_url or os.environ.get("RAG_RUNNER_URL") or DEFAULT_RUNNER_URL
            ),
            image=image or os.environ.get("RAG_PARSER_IMAGE") or DEFAULT_PARSER_IMAGE,
            task_command=tuple(command),
            task_env=env,
            cpu=cpu if cpu is not None else _float_env("RAG_TASK_CPU"),
            memory_mb=(
                memory_mb if memory_mb is not None else _int_env("RAG_TASK_MEMORY_MB")
            ),
            max_in_flight=(
                max_in_flight
                if max_in_flight is not None
                else _int_env("RAG_MAX_IN_FLIGHT") or DEFAULT_MAX_IN_FLIGHT
            ),
            batch_size=(
                batch_size
                if batch_size is not None
                else _int_env("RAG_DISPATCH_BATCH") or DEFAULT_BATCH_SIZE
            ),
            ack_on=ack,
            task_timeout=(
                task_timeout
                if task_timeout is not None
                else _float_env("RAG_TASK_TIMEOUT") or DEFAULT_TASK_TIMEOUT
            ),
        )


def _env_pairs(raw: str | None) -> list[tuple[str, str]]:
    """`"A=1,B=2"` -> [("A", "1"), ("B", "2")]. Empty and None mean nothing."""
    pairs = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        pairs.append((key.strip(), value))
    return pairs


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
        """Build settings from the environment; `None` keywords mean "not passed"."""
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
