"""
The parse cache: one JSON per content hash, plus a sidecar manifest.

The manifest exists because the two steps run as separate processes. Step 1
knows the uri and the mime type; step 2 only gets a sha. Rather than making
step 2 re-derive that from the source (which may no longer be reachable), step 1
writes it next to the parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

log = logging.getLogger(__name__)


def sha256_of(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def cache_path(sha: str, cache_dir: Path) -> Path:
    return cache_dir / f"{sha}.json"


def manifest_path(sha: str, cache_dir: Path) -> Path:
    return cache_dir / f"{sha}.meta.json"


@dataclass(frozen=True)
class Manifest:
    """What step 2 needs to know about a parse it did not perform."""

    sha256: str
    uri: str
    mime: str
    parser_version: str
    parsed_at: str

    def write(self, cache_dir: Path) -> Path:
        path = manifest_path(self.sha256, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @staticmethod
    def read(sha: str, cache_dir: Path) -> "Manifest":
        path = manifest_path(sha, cache_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"no manifest for {sha}; re-run step 1 for this document"
            )
        return Manifest(**json.loads(path.read_text()))

    @staticmethod
    def now(sha: str, uri: str, mime: str, parser_version: str) -> "Manifest":
        return Manifest(
            sha256=sha,
            uri=uri,
            mime=mime,
            parser_version=parser_version,
            parsed_at=datetime.now(timezone.utc).isoformat(),
        )


def parse_and_cache(
    uri: str,
    blob: bytes,
    cache_dir: Path,
    *,
    source: str | None = None,
) -> tuple[str, DoclingDocument]:
    """Parse once. Write the JSON before deriving anything, so a crash in a
    downstream branch never costs a re-parse on retry.

    `source` is what Docling actually opens — a local path when `uri` names a
    remote object whose bytes were staged to disk. It defaults to `uri`.
    """
    sha = sha256_of(blob)
    path = cache_path(sha, cache_dir)

    if path.exists():
        log.info("cache hit %s", sha[:12])
        return sha, DoclingDocument.load_from_json(path)

    doc = DocumentConverter().convert(source or uri).document
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(path)
    log.info("parsed and cached %s (%s)", sha[:12], uri)
    return sha, doc


def load_cached(sha: str, cache_dir: Path) -> DoclingDocument:
    path = cache_path(sha, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"no cached parse for {sha}; re-run step 1")
    return DoclingDocument.load_from_json(path)


def cached_shas(cache_dir: Path) -> list[str]:
    """Every sha with both a parse and a manifest, in a stable order."""
    if not cache_dir.exists():
        return []
    return sorted(
        p.stem
        for p in cache_dir.glob("*.json")
        if not p.name.endswith(".meta.json")
        and manifest_path(p.stem, cache_dir).exists()
    )
