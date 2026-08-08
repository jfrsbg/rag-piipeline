"""The two independently runnable halves of the pipeline."""

from .parse import parse_documents
from .index import index_documents

__all__ = ["parse_documents", "index_documents"]
