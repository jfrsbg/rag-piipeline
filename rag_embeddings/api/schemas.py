"""
Request and response models for the search API.

These mirror the dicts `steps.search` returns, not the `chunks` table.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="the question, in the corpus's language",
        examples=["what were the Q3 logistics costs?"],
    )
    limit: int = Field(5, ge=1, le=50, description="hits to return")
    window: int = Field(
        0,
        ge=0,
        le=5,
        description="also return this many chunks either side of each hit",
    )


class WindowChunk(BaseModel):
    ord: int
    text: str


class Hit(BaseModel):
    id: int
    document_id: int
    ord: int
    page: int | None = None
    text: str
    heading_path: list[str] = []
    uri: str
    chunk_config: str
    # Cosine distance: smaller is closer. Not a similarity score, and not
    # normalised to 0–1, because the ordering is the answer and rescaling it
    # would invent a confidence the index does not have.
    distance: float
    # This hit was indexed under a different profile than the query was
    # embedded with, so its distance is not comparable to the others'.
    stale: bool
    # Absent unless `window` was asked for.
    window: list[WindowChunk] | None = None


class SearchResponse(BaseModel):
    query: str
    # The profile the query was embedded with. Any hit whose chunk_config
    # differs from this is the mismatch, and is flagged `stale`.
    profile: str
    count: int
    hits: list[Hit]
