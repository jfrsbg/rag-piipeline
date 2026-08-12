"""
The pipeline's write path: a producer and two pools of long-lived consumers.

    enqueue.files  ->  to-parse  ->  parse_worker  ->  to-index  ->  index_worker

There is no batch counterpart. Both consumers are services that come up with the
deployment and stay up, blocking on an empty queue rather than exiting when it
drains — a backlog is a normal state, not a finish line. Everything a
container-per-document would rebuild each time (a resident model, a database
connection) is built once before the loop, and the loop itself holds nothing.

The names below resolve on first access, for the same reason the top-level
package does it: importing this package eagerly meant the producer imported
`index_worker`, which imports `embedder`, which imports torch — several seconds
and a few thousand modules to put a filename on a queue, paid by the smallest
and most-replicated container in the fan-out. Importing a submodule directly
(`from rag_embeddings.workers import enqueue`) is unaffected either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# name -> the submodule that defines it
_EXPORTS = {
    "enqueue_files": ".enqueue",
    "enqueue_cached": ".enqueue",
    "enqueue_stale": ".enqueue",
    "run_parse_worker": ".parse_worker",
    "run_index_worker": ".index_worker",
}

# The two consumers export their loop under one name; `run` is what the module
# calls it, and the pool it belongs to is what the caller cares about.
_ALIASES = {"run_parse_worker": "run", "run_index_worker": "run"}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """PEP 562: resolve an export by importing only what defines it."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, _ALIASES.get(name, name))
    globals()[name] = value                     # import once, then it is normal
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:                               # pragma: no cover
    from .enqueue import enqueue_cached, enqueue_files, enqueue_stale
    from .index_worker import run as run_index_worker
    from .parse_worker import run as run_parse_worker
