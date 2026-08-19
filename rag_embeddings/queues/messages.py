"""
Message payloads: one frozen dataclass per queue, each with an explicit
`from_body` so a message written by an older deploy fails loudly at the edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:                                    # pragma: no cover
    from ..cache import Manifest

# Schemes that name a file on whichever machine is reading the message. Bare
# paths have no scheme at all, which is what `enqueue files` writes.
LOCAL_SCHEMES = ("", "file")


def local_path(uri: str) -> Path | None:
    """Return the local file a uri names, or None if the uri is remote."""
    parsed = urlparse(uri)
    if parsed.scheme not in LOCAL_SCHEMES or parsed.netloc:
        return None
    # A bare path is not a url and must not be unquoted: "Q3%20report.pdf" is a
    # real filename. Only file:// carries percent-encoding.
    return Path(unquote(parsed.path) if parsed.scheme else uri)


@dataclass(frozen=True)
class ParseRequest:
    """Step 1's unit of work: one document to parse."""

    uri: str
    mime: str | None = None
    uri_prefix: str | None = None
    force: bool = False

    def to_body(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> "ParseRequest":
        return cls(
            uri=body["uri"],
            mime=body.get("mime"),
            uri_prefix=body.get("uri_prefix"),
            force=bool(body.get("force", False)),
        )


@dataclass(frozen=True)
class IndexRequest:
    """Step 2's unit of work: one cached parse to index."""

    sha256: str
    uri: str
    mime: str
    parser_version: str
    parsed_at: str
    with_tables: bool = True
    with_chunks: bool = True

    def to_body(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> "IndexRequest":
        return cls(
            sha256=body["sha256"],
            uri=body["uri"],
            mime=body["mime"],
            parser_version=body["parser_version"],
            parsed_at=body["parsed_at"],
            with_tables=bool(body.get("with_tables", True)),
            with_chunks=bool(body.get("with_chunks", True)),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: "Manifest",
        *,
        with_tables: bool = True,
        with_chunks: bool = True,
    ) -> "IndexRequest":
        return cls(
            sha256=manifest.sha256,
            uri=manifest.uri,
            mime=manifest.mime,
            parser_version=manifest.parser_version,
            parsed_at=manifest.parsed_at,
            with_tables=with_tables,
            with_chunks=with_chunks,
        )

    def to_manifest(self) -> "Manifest":
        """Rebuild the cache sidecar this message stands in for."""
        from ..cache import Manifest

        return Manifest(
            sha256=self.sha256,
            uri=self.uri,
            mime=self.mime,
            parser_version=self.parser_version,
            parsed_at=self.parsed_at,
        )
