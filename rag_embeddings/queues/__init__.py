"""
Queue backends and `open_queue`, which picks one from a uri.
The uri is the only place a backend is named.
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
from .messages import IndexRequest, ParseRequest, local_path

DEFAULT_QUEUE_URL = "file://./queue"


def open_queue(url: str | Path, name: str, **kwargs) -> Queue:
    """Build the queue named `name` at `url` (`memory://`, `file://`, or a path)."""
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
    "local_path",
    "open_queue",
    "shared_memory",
    "reset_memory",
    "DEFAULT_QUEUE_URL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_VISIBILITY_TIMEOUT",
]
