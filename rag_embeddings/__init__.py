"""
Document ingestion: parse once, cache, fan out to tables (relational)
and chunks (vector), commit both in one transaction.

The pipeline is split into two independently runnable steps:

    step 1  parse and cache      rag_embeddings.steps.parse
    step 2  extract and store    rag_embeddings.steps.index

Step 1 never touches the database; step 2 never touches a parser.
"""

from .profiles import EmbedProfile, BGE_M3, E5_LARGE, PROFILES
from .embedder import Embedder
from .config import Settings
from .cache import parse_and_cache, load_cached, cache_path
from .extraction.tables import extract_tables
from .extraction.chunks import build_chunks
from .storage.connection import connect
from .storage.writer import write_all
from .pipeline import ingest, reextract, rechunk, stale_documents

__all__ = [
    "EmbedProfile",
    "BGE_M3",
    "E5_LARGE",
    "PROFILES",
    "Embedder",
    "Settings",
    "parse_and_cache",
    "load_cached",
    "cache_path",
    "extract_tables",
    "build_chunks",
    "connect",
    "write_all",
    "ingest",
    "reextract",
    "rechunk",
    "stale_documents",
]
