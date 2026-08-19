"""Document ingestion: parse once, cache, fan out to tables and chunks.

Exports resolve lazily (PEP 562) so importing a submodule such as `queues`
never pulls in docling and torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# name -> the submodule that defines it
_EXPORTS = {
    "EmbedProfile": ".profiles",
    "BGE_M3": ".profiles",
    "E5_LARGE": ".profiles",
    "PROFILES": ".profiles",
    "Embedder": ".embedder",
    "Settings": ".config",
    "BlobStore": ".blobstore",
    "LocalBlobStore": ".blobstore",
    "open_store": ".blobstore",
    "parse_and_cache": ".cache",
    "load_cached": ".cache",
    "cache_path": ".cache",
    "drop_cached": ".cache",
    "extract_tables": ".extraction.tables",
    "build_chunks": ".extraction.chunks",
    "connect": ".storage.connection",
    "write_all": ".storage.writer",
    "ingest": ".pipeline",
    "reextract": ".pipeline",
    "rechunk": ".pipeline",
    "stale_documents": ".pipeline",
    "Queue": ".queues",
    "ParseRequest": ".queues",
    "IndexRequest": ".queues",
    "open_queue": ".queues",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve an export by importing only the submodule that defines it."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value                     # import once, then it is normal
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:                               # pragma: no cover
    # Lazy exports are invisible to type checkers and editors, so the real
    # imports are declared here where only they will read them.
    from .blobstore import BlobStore, LocalBlobStore, open_store
    from .cache import cache_path, drop_cached, load_cached, parse_and_cache
    from .config import Settings
    from .embedder import Embedder
    from .extraction.chunks import build_chunks
    from .extraction.tables import extract_tables
    from .pipeline import ingest, rechunk, reextract, stale_documents
    from .profiles import BGE_M3, E5_LARGE, PROFILES, EmbedProfile
    from .queues import IndexRequest, ParseRequest, Queue, open_queue
    from .storage.connection import connect
    from .storage.writer import write_all
