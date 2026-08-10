"""
Queues, and the one function that chooses between them.

    from rag_embeddings.queues import open_queue
    q = open_queue("file:///queue", "to-parse")

The uri is the only place a backend is named. Everything else — producer,
both workers, the tests — takes a Queue and does not care which one it got,
which is what makes `RAG_QUEUE_URL=sqs://...` the whole migration.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .base import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_VISIBILITY_TIMEOUT,
    ConsumeStats,
    Message,
    Queue,
)
from .files import FileQueue
from .memory import InMemoryQueue, reset as reset_memory, shared as shared_memory
from .messages import IndexRequest, ParseRequest

DEFAULT_QUEUE_URL = "file://./queue"


def open_queue(url: str | Path, name: str, **kwargs) -> Queue:
    """Build the queue named `name` at `url`.

    `memory://` is process-local and returns the same object for the same
    name, so a producer and a consumer in one test see one queue. `file://`
    and bare paths are directories, which is what lets separate containers
    sharing a volume see one queue.

    A broker backend is a Queue subclass implementing the six transport
    methods plus one branch here; the retry and dead-letter behaviour it
    inherits from `base` is already the behaviour the workers were tested
    against.
    """
    if isinstance(url, Path):
        return FileQueue(url, name, **kwargs)

    parsed = urlparse(str(url))
    if parsed.scheme == "memory":
        return shared_memory(name, **kwargs)
    if parsed.scheme in ("", "file"):
        # urlparse puts a relative path in .path but a bare "./queue" has no
        # scheme at all, so fall back to the raw string.
        root = f"{parsed.netloc}{parsed.path}" if parsed.scheme else str(url)
        return FileQueue(root, name, **kwargs)
    raise NotImplementedError(
        f"no queue backend for {parsed.scheme!r}:// — subclass Queue and add a "
        f"branch to open_queue()"
    )


__all__ = [
    "Queue",
    "Message",
    "ConsumeStats",
    "FileQueue",
    "InMemoryQueue",
    "ParseRequest",
    "IndexRequest",
    "open_queue",
    "shared_memory",
    "reset_memory",
    "DEFAULT_QUEUE_URL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_VISIBILITY_TIMEOUT",
]
