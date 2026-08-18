"""
The pipeline's write path: a producer, a dispatcher, and one long-lived pool.

    enqueue.files -> to-parse -> dispatcher -> parse_worker (a job, per document)
                                                   |
                                                   v
                                               to-index -> index_worker (a pool)

The two steps are deliberately different shapes, because the thing that decides
the shape is what a container has to build before it can do any work.

Step 1 builds nothing worth keeping, so it is a job. The dispatcher consumes
`to-parse` and starts one `parse_worker` container per document, passing it the
message it just claimed; that container parses its document, publishes onto
`to-index` and exits. Nothing waits on an empty queue, and the number of
parsers tracks the backlog rather than the deployment.

Step 2 is a pool of services for the opposite reason: a resident embedding
model and a database connection are exactly what you cannot afford to rebuild
per document, so `index_worker` comes up with the deployment, blocks on an
empty `to-index` and stays up when it drains.

Both halves are the same messages and the same guarantees either way — a claim
is held until the work is accounted for, and the queue owns retries.

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
    "run_dispatcher": ".dispatcher",
    "Dispatcher": ".dispatcher",
}

# Each module calls its entrypoint `run`, and which worker it belongs to is
# what the caller cares about. `run_parse_worker` is the odd one: it takes a
# message body and parses one document, where the other two take settings and
# loop — the difference between a job and a service, at the seam.
_ALIASES = {
    "run_parse_worker": "run",
    "run_index_worker": "run",
    "run_dispatcher": "run",
}

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
    from .dispatcher import Dispatcher
    from .dispatcher import run as run_dispatcher
    from .enqueue import enqueue_cached, enqueue_files, enqueue_stale
    from .index_worker import run as run_index_worker
    from .parse_worker import run as run_parse_worker
