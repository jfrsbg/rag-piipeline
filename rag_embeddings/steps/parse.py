"""
Source resolution: what a source string means, decided in one place.

Stdlib-only by design; the parse itself is `cache.parse_and_cache`.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable

DEFAULT_MIME = "application/octet-stream"


def resolve_sources(sources: Iterable[str], pattern: str = "*") -> list[Path]:
    """Expand files, directories and globs into a stable, deduplicated list."""
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
    """Guess the mime type to record for a document, by name alone."""
    if override:
        return override
    return mimetypes.guess_type(Path(name).name)[0] or DEFAULT_MIME
