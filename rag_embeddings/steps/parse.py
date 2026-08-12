"""
Source resolution — what a source string means, decided in one place.

Turning `inbox/`, `'inbox/*.pdf'` or an explicit path into a concrete list of
files is the producer's job and nobody else's. It enumerates once, publishes
one message per document, and every parse container downstream is then handed a
single uri it never has to expand. That asymmetry is the whole reason the pool
scales: a worker that enumerates is a worker that has an opinion about how much
work exists, and thirty of those racing over the same directory is a scan, not
a pipeline.

The parse itself is `cache.parse_and_cache`. The service that calls it once per
message is `rag_embeddings.workers.parse_worker`. Neither is imported here, and
that is deliberate: this module is stdlib-only, so resolving a directory costs
nothing on its own. (The producer still pays for docling today, because it
imports `cache` for `Manifest` and `cached_shas` and `cache` imports
`DocumentConverter` at module scope — that import is the remaining cost, not
this file.)
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable

DEFAULT_MIME = "application/octet-stream"


def resolve_sources(sources: Iterable[str], pattern: str = "*") -> list[Path]:
    """Expand files, directories and globs into a stable list of files.

    A directory expands by `pattern`; anything else is passed to Path.glob on
    its parent so shell-unexpanded patterns still work inside a container.
    """
    found: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            found.extend(p for p in sorted(path.glob(pattern)) if p.is_file())
        elif path.exists():
            found.append(path)
        else:
            matches = sorted(Path(path.parent or ".").glob(path.name))
            if not matches:
                raise FileNotFoundError(f"no source matched {raw!r}")
            found.extend(p for p in matches if p.is_file())

    # Same file named twice (a dir and an explicit path) should enqueue once.
    return list(dict.fromkeys(found))


def guess_mime(name: str | Path, override: str | None = None) -> str:
    """The mime type to record for a document.

    Takes a name rather than an open file: the producer has a path and the
    worker has a uri, and both must arrive at the same answer for the same
    document — the mime on the row is the producer's message when it set one.
    """
    if override:
        return override
    return mimetypes.guess_type(Path(name).name)[0] or DEFAULT_MIME
