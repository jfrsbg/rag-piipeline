"""
Step 1 — parse and cache.

Reads source documents, parses each one once, and writes the parse plus a
manifest into the cache. No database, no embedding model, no GPU: this step
only needs Docling and disk, which is why it is worth running on its own.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
from pathlib import Path
from typing import Iterable, Sequence

from ..cache import Manifest, drop_cached, parse_and_cache, sha256_of
from ..cli import common_parser, configure_logging, settings_from
from ..config import Settings

log = logging.getLogger(__name__)

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

    # Same file named twice (a dir and an explicit path) should parse once.
    return list(dict.fromkeys(found))


def guess_mime(path: Path, override: str | None) -> str:
    if override:
        return override
    return mimetypes.guess_type(path.name)[0] or DEFAULT_MIME


def parse_documents(
    sources: Sequence[str],
    settings: Settings | None = None,
    *,
    mime: str | None = None,
    pattern: str = "*",
    uri_prefix: str | None = None,
    force: bool = False,
) -> list[Manifest]:
    """Parse every source and cache it. Returns one manifest per document.

    `uri_prefix` records a different location than the one being read — use it
    when the bytes were staged from object storage and the s3:// uri is what
    should end up in `documents.uri`.
    """
    settings = settings or Settings.from_env()
    paths = resolve_sources(sources, pattern)
    log.info("step 1: %d source(s) -> %s", len(paths), settings.cache_dir)

    manifests: list[Manifest] = []
    for path in paths:
        blob = path.read_bytes()
        uri = f"{uri_prefix.rstrip('/')}/{path.name}" if uri_prefix else str(path)

        if force:
            drop_cached(sha256_of(blob), settings.cache_dir)

        sha, _doc = parse_and_cache(uri, blob, settings.cache_dir, source=str(path))
        manifest = Manifest.now(
            sha, uri, guess_mime(path, mime), settings.parser_version
        )
        manifest.write(settings.cache_dir)
        manifests.append(manifest)

    log.info("step 1 done: %d document(s) cached", len(manifests))
    return manifests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="step1-parse",
        parents=[common_parser()],
        description="Step 1: parse documents and write them to the parse cache.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="files, directories or globs to parse (e.g. inbox/ or 'inbox/*.pdf')",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="glob applied when a source is a directory (default: %(default)s)",
    )
    parser.add_argument("--mime", help="override the guessed mime type")
    parser.add_argument(
        "--uri-prefix",
        help="record uris under this prefix instead of the local path, "
             "e.g. s3://bucket/key",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="drop the cached parse first (Docling upgrade, OCR settings change)",
    )

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    manifests = parse_documents(
        args.sources,
        settings_from(args),
        mime=args.mime,
        pattern=args.pattern,
        uri_prefix=args.uri_prefix,
        force=args.force,
    )
    for m in manifests:
        print(f"{m.sha256}\t{m.uri}")
    return 0
