"""
Document ingestion: parse once, cache, fan out to tables (relational)
and chunks (vector), commit both in one transaction.

Entry points
------------
ingest(uri)      parse + cache + branch A + branch B
reextract(sha)   cache -> branch A only   (schema changed)
rechunk(sha)     cache -> branch B only   (embedding model changed)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/parsed")
PARSER_VERSION = "docling-2.91"


# --------------------------------------------------------------------------
# Profile: tokenizer, chunker and embedder all derive from one object so a
# mismatch requires constructing two profiles, which is visible in a diff.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedProfile:
    model_id: str
    max_tokens: int          # from the model card, never model_max_length
    headroom: int = 128      # contextualize() is applied after the split
    passage_prefix: str = ""
    query_prefix: str = ""

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.max_tokens}-{self.headroom}"


BGE_M3 = EmbedProfile(model_id="BAAI/bge-m3", max_tokens=8192)


class Embedder:
    def __init__(self, profile: EmbedProfile):
        self.profile = profile
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(profile.model_id),
            max_tokens=profile.max_tokens - profile.headroom,
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=True)
        self.model = SentenceTransformer(profile.model_id)

    def count(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.profile.passage_prefix + t for t in texts]
        vecs = self.model.encode(prefixed, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [self.profile.query_prefix + text], normalize_embeddings=True
        )[0]
        return vec.tolist()


# --------------------------------------------------------------------------
# Step 1: parse and cache
# --------------------------------------------------------------------------

def cache_path(sha: str) -> Path:
    return CACHE_DIR / f"{sha}.json"


def parse_and_cache(uri: str, blob: bytes) -> tuple[str, DoclingDocument]:
    """Parse once. Write the JSON before deriving anything, so a crash in a
    downstream branch never costs a re-parse on retry."""
    sha = hashlib.sha256(blob).hexdigest()
    path = cache_path(sha)

    if path.exists():
        log.info("cache hit %s", sha[:12])
        return sha, DoclingDocument.load_from_json(path)

    doc = DocumentConverter().convert(uri).document
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(path)
    log.info("parsed and cached %s (%s)", sha[:12], uri)
    return sha, doc


def load_cached(sha: str) -> DoclingDocument:
    path = cache_path(sha)
    if not path.exists():
        raise FileNotFoundError(f"no cached parse for {sha}; re-run ingest()")
    return DoclingDocument.load_from_json(path)


# --------------------------------------------------------------------------
# Branch A: tables -> relational
# --------------------------------------------------------------------------

def _first_prov(item: Any):
    """prov is empty on merged groups and some HTML-derived nodes."""
    return next(iter(getattr(item, "prov", []) or []), None)


def extract_tables(doc: DoclingDocument, document_id: int) -> list[dict]:
    """Reads table_cells, not export_to_dataframe(): the DataFrame export has a
    known class of bug where a column is silently dropped, while the cell data
    is intact in the JSON."""
    rows = []
    for idx, table in enumerate(doc.tables):
        data = table.data
        cells = [
            {
                "text": c.text,
                "row": c.start_row_offset_idx,
                "row_end": c.end_row_offset_idx,
                "col": c.start_col_offset_idx,
                "col_end": c.end_col_offset_idx,
                "is_col_header": bool(getattr(c, "column_header", False)),
                "is_row_header": bool(getattr(c, "row_header", False)),
            }
            for c in (data.table_cells or [])
        ]
        headers = [c["text"] for c in cells if c["is_col_header"]]
        prov = _first_prov(table)

        try:
            caption = table.caption_text(doc)
        except Exception:
            caption = None

        rows.append({
            "document_id": document_id,
            "table_index": idx,
            "self_ref": table.self_ref,          # join key to the chunk copy
            "page": prov.page_no if prov else None,
            "caption": caption,
            "num_rows": getattr(data, "num_rows", None),
            "num_cols": getattr(data, "num_cols", None),
            "columns": Jsonb(headers),
            "cells": Jsonb(cells),
            "markdown": table.export_to_markdown(doc),
            "parser_version": PARSER_VERSION,
        })
    return rows


# --------------------------------------------------------------------------
# Branch B: chunks -> vectors
# --------------------------------------------------------------------------

def build_chunks(doc: DoclingDocument, document_id: int, emb: Embedder) -> list[dict]:
    rows: list[dict] = []
    to_embed: list[str] = []

    # Consume the iterator once, in order: emission order is reading order,
    # which is the only thing that makes `ord` meaningful.
    for ord_, chunk in enumerate(emb.chunker.chunk(dl_doc=doc)):
        prov = next(
            (p for it in chunk.meta.doc_items for p in (it.prov or [])), None
        )
        enriched = emb.chunker.contextualize(chunk=chunk)

        n = emb.count(enriched)
        if n > emb.profile.max_tokens:
            # Enrichment is added after the split decision, so wide tables and
            # deep heading paths can still overflow. Record it; do not truncate
            # silently.
            log.warning(
                "chunk overflow doc=%s ord=%s tokens=%s limit=%s",
                document_id, ord_, n, emb.profile.max_tokens,
            )

        rows.append({
            "document_id": document_id,
            "ord": ord_,
            "text": chunk.text,
            "heading_path": list(chunk.meta.headings or []),
            "page": prov.page_no if prov else None,
            "refs": Jsonb([it.self_ref for it in chunk.meta.doc_items]),
            "token_count": n,
            "embed_model": emb.profile.model_id,
            "chunk_config": emb.profile.version,
        })
        to_embed.append(enriched)

    # One batched call; encode() preserves input order, and rows/to_embed were
    # appended in lockstep, so zip is safe.
    for row, vec in zip(rows, emb.encode_passages(to_embed)):
        row["embedding"] = vec

    return rows


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

DOC_UPSERT = """
insert into documents (sha256, uri, mime, parser_version, parsed_at, status)
values (%(sha256)s, %(uri)s, %(mime)s, %(parser_version)s, now(), 'ok')
on conflict (sha256) do update set uri = excluded.uri, parsed_at = now()
returning id
"""

TABLE_INSERT = """
insert into doc_tables (document_id, table_index, self_ref, page, caption,
                        num_rows, num_cols, columns, cells, markdown, parser_version)
values (%(document_id)s, %(table_index)s, %(self_ref)s, %(page)s, %(caption)s,
        %(num_rows)s, %(num_cols)s, %(columns)s, %(cells)s, %(markdown)s,
        %(parser_version)s)
"""

CHUNK_INSERT = """
insert into chunks (document_id, ord, text, heading_path, page, refs,
                    embedding, embed_model, chunk_config, token_count)
values (%(document_id)s, %(ord)s, %(text)s, %(heading_path)s, %(page)s, %(refs)s,
        %(embedding)s, %(embed_model)s, %(chunk_config)s, %(token_count)s)
"""


def write_all(conn, sha: str, uri: str, mime: str,
              tables: list[dict] | None, chunks: list[dict] | None) -> None:
    """Both branches in one commit. Otherwise a failure between them leaves a
    document with vectors but no rows, and nothing knows it is half-ingested."""
    with conn.transaction():
        cur = conn.cursor()
        cur.execute(DOC_UPSERT, {
            "sha256": sha, "uri": uri, "mime": mime,
            "parser_version": PARSER_VERSION,
        })
        document_id = cur.fetchone()[0]

        if tables is not None:
            cur.execute("delete from doc_tables where document_id = %s", (document_id,))
            for r in tables:
                r["document_id"] = document_id
            cur.executemany(TABLE_INSERT, tables)

        if chunks is not None:
            cur.execute("delete from chunks where document_id = %s", (document_id,))
            for r in chunks:
                r["document_id"] = document_id
            cur.executemany(CHUNK_INSERT, chunks)

    log.info(
        "wrote %s: %d tables, %d chunks",
        sha[:12], len(tables or []), len(chunks or []),
    )


def _document_id(conn, sha: str) -> int:
    row = conn.execute("select id from documents where sha256 = %s", (sha,)).fetchone()
    if row is None:
        raise LookupError(f"unknown document {sha}")
    return row[0]


# --------------------------------------------------------------------------
# Entry points — same cache, three reprocessing triggers
# --------------------------------------------------------------------------

def ingest(conn, uri: str, blob: bytes, mime: str, emb: Embedder) -> str:
    sha = hashlib.sha256(blob).hexdigest()
    if conn.execute("select 1 from documents where sha256 = %s", (sha,)).fetchone():
        log.info("skip, already ingested %s", sha[:12])
        return sha

    sha, doc = parse_and_cache(uri, blob)
    tables = extract_tables(doc, document_id=0)
    chunks = build_chunks(doc, document_id=0, emb=emb)
    write_all(conn, sha, uri, mime, tables, chunks)
    return sha


def reextract(conn, sha: str) -> None:
    """Extraction schema changed. Branch A only — no GPU, no re-parse."""
    doc = load_cached(sha)
    document_id = _document_id(conn, sha)
    tables = extract_tables(doc, document_id)
    with conn.transaction():
        conn.execute("delete from doc_tables where document_id = %s", (document_id,))
        conn.cursor().executemany(TABLE_INSERT, tables)


def rechunk(conn, sha: str, emb: Embedder) -> None:
    """Embedding model or chunk config changed. Branch B only."""
    doc = load_cached(sha)
    document_id = _document_id(conn, sha)
    chunks = build_chunks(doc, document_id, emb)
    with conn.transaction():
        conn.execute("delete from chunks where document_id = %s", (document_id,))
        conn.cursor().executemany(CHUNK_INSERT, chunks)


def stale_documents(conn, emb: Embedder) -> list[str]:
    """Which documents need rechunk() after a profile change."""
    rows = conn.execute(
        """select distinct d.sha256 from documents d
           join chunks c on c.document_id = d.id
           where c.chunk_config <> %s""",
        (emb.profile.version,),
    ).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------

def connect(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    emb = Embedder(BGE_M3)
    conn = connect("postgresql://localhost/docs")

    for path in Path("inbox").glob("*"):
        ingest(conn, str(path), path.read_bytes(), mime="application/pdf", emb=emb)