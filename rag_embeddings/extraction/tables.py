"""Branch A: tables -> relational."""

from __future__ import annotations

from typing import Any

from docling_core.types.doc import DoclingDocument
from psycopg.types.json import Jsonb


def _first_prov(item: Any):
    """prov is empty on merged groups and some HTML-derived nodes."""
    return next(iter(getattr(item, "prov", []) or []), None)


def extract_tables(
    doc: DoclingDocument,
    document_id: int,
    parser_version: str,
) -> list[dict]:
    """Extract one row per table from the parsed document."""
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
            "parser_version": parser_version,
        })
    return rows
