"""
Document ingestion: parse once, cache, fan out to tables (relational)
and chunks (vector), commit both in one transaction.

The pipeline is two pools of services with a queue between them:

    step 1  parse and cache      rag_embeddings.workers.parse_worker
    step 2  extract and store    rag_embeddings.workers.index_worker

Step 1 never touches the database; step 2 never touches a parser. Both are fed
one document per message by `workers.enqueue`, and both stay up across an empty
queue — there is no batch driver that walks a directory, because a process that
enumerates its own work cannot be replicated. The seams that makes possible are
`queues` (which broker) and `blobstore` (where the shared cache lives); nothing
above either one names a backend.

`pipeline` is the in-process library API — one document, one call, no queue. It
is for embedding this in something else, not for running the pipeline.

The names below are resolved on first access rather than on import. Eagerly
importing them would mean `import rag_embeddings.queues` — all a producer
needs to put a filename on a queue — paid for docling and torch on the way in,
and that cost lands on the smallest, most-replicated container in the fan-out.
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
    """PEP 562: resolve an export by importing only what defines it."""
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
