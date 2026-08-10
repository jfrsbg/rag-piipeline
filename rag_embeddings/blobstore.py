"""
Where the parse cache lives.

Step 1 writes a parse and step 2 reads it, and once those are separate
containers the bind mount that made that work stops being an answer. So every
touch of the cache goes through a BlobStore and the deployment chooses the
backend: a shared directory on one machine, object storage once the workers no
longer share a filesystem.

Docling's API is path-oriented — `save_as_json` and `load_from_json` both want
a real file — so the interface is a pair of context managers yielding a path,
not a bytes-in/bytes-out pair. Locally the yielded path *is* the cache file and
nothing is copied; a remote backend stages a temp file and transfers around the
yield. Making the local case pay a copy to satisfy the remote one would be the
wrong trade: the local case is every test run and every laptop.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


class BlobStore(ABC):
    """Keys in, bytes out. Nothing above this knows where the cache lives."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Missing keys are not an error — callers use this to clear a slot."""

    @abstractmethod
    def keys(self, suffix: str = "") -> list[str]:
        """Every key, sorted, optionally filtered by suffix."""

    @abstractmethod
    @contextmanager
    def reading(self, key: str) -> Iterator[Path]:
        """A local path holding `key`, valid for the duration of the block."""

    @abstractmethod
    @contextmanager
    def writing(self, key: str) -> Iterator[Path]:
        """A local path to write `key` into; stored when the block exits.

        A block that raises stores nothing, so a crashed parse never leaves a
        half-written entry that the next run would treat as a cache hit.
        """


class LocalBlobStore(BlobStore):
    """A directory. The yielded paths are the real files, so there is no copy."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"LocalBlobStore({str(self.root)!r})"

    def path_for(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, data)

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def keys(self, suffix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_file() and p.name.endswith(suffix)
        )

    @contextmanager
    def reading(self, key: str) -> Iterator[Path]:
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"{key} not in {self.root}")
        yield path

    @contextmanager
    def writing(self, key: str) -> Iterator[Path]:
        final = self.path_for(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        # Same directory as the target so the rename below stays on one
        # filesystem, and prefixed so a reaper can tell scratch from cache.
        tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
        try:
            yield tmp
            if not tmp.exists():
                raise RuntimeError(f"nothing was written to {tmp}")
            os.replace(tmp, final)
        finally:
            tmp.unlink(missing_ok=True)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a sibling temp file, so a reader never sees a partial file.

    Two workers handed the same document race here, and both are writing the
    same bytes — the rename decides which one wins and the loser's copy is
    discarded whole. That is the property worth having: never a torn file.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def open_store(target: BlobStore | Path | str) -> BlobStore:
    """Resolve a cache location to a store.

    A BlobStore passes through, which is what lets callers keep taking a plain
    directory while a worker injects something else. Plain paths and `file://`
    are local; the scheme is where a remote backend hooks in.

    Adding S3 is a class implementing the five primitives plus the two context
    managers (download to a NamedTemporaryFile in `reading`, upload after the
    yield in `writing`) and one more branch here. Nothing above this changes.
    """
    if isinstance(target, BlobStore):
        return target
    if isinstance(target, Path):
        return LocalBlobStore(target)

    parsed = urlparse(str(target))
    if parsed.scheme in ("", "file"):
        return LocalBlobStore(parsed.path if parsed.scheme else str(target))
    raise NotImplementedError(
        f"no store backend for {parsed.scheme!r}:// — implement BlobStore for it "
        f"and add a branch to open_store()"
    )


def copy_tree(src: BlobStore, dst: BlobStore, suffix: str = "") -> int:
    """Move a cache between backends — local to S3 on the way to a cluster."""
    moved = 0
    for key in src.keys(suffix):
        dst.write_bytes(key, src.read_bytes(key))
        moved += 1
    return moved


__all__ = ["BlobStore", "LocalBlobStore", "open_store", "copy_tree"]
