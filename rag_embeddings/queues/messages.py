"""
What goes on the wire.

Two message types, one per queue, each a frozen dataclass with an explicit
`from_body` — because the thing on the other end of a queue is JSON written by
a version of this code you are no longer running. Parsing it into a dataclass
at the edge means a rolling deploy fails on the first message with a clear
KeyError instead of somewhere deep in the pipeline with a None.

IndexRequest is a Manifest plus the branch flags, and that is deliberate: step
2's whole reason for reading a sidecar file was that it received only a sha.
Given a queue, the message carries what the sidecar carried, and the read goes
away. The sidecar is still written, because the single-machine path has no
queue to carry anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:                                    # pragma: no cover
    from ..cache import Manifest


@dataclass(frozen=True)
class ParseRequest:
    """Step 1's unit of work: one document, wherever its bytes are."""

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
    """Step 2's unit of work: one cached parse, described well enough to store.

    Never a list of shas and never "everything in the cache" — a worker is told
    which document it owns. Enumerating the cache is a coordinator's job.
    """

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
        """The sidecar step 2 would otherwise have read off the cache.

        Imported here rather than at module scope so that publishing a message
        does not require the parser: `cache` pulls in docling, and the producer
        — an S3 event handler, a cron, a shell loop — has no business carrying
        a 2 GB dependency to write a filename onto a queue.
        """
        from ..cache import Manifest

        return Manifest(
            sha256=self.sha256,
            uri=self.uri,
            mime=self.mime,
            parser_version=self.parser_version,
            parsed_at=self.parsed_at,
        )
