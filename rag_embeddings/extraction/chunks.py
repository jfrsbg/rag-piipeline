"""Branch B: chunks -> vectors."""

from __future__ import annotations

import logging

from docling_core.types.doc import DoclingDocument
from psycopg.types.json import Jsonb

from ..embedder import Embedder

log = logging.getLogger(__name__)


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
