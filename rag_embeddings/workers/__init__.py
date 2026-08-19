"""
The write path: enqueue -> to-parse -> dispatcher -> parse_worker -> to-index
-> index_worker. Exports resolve on first access so that importing this package
does not drag in `index_worker` and, through it, torch.
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
