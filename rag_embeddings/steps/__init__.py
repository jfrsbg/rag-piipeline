"""
What survives the write path being queue-only.

`parse` is source resolution — the producer's half of step 1, kept apart from
the parse itself so it costs nothing to import. `search` is the read path the
API serves. The work is in `rag_embeddings.workers`, one document per message.
"""

from .parse import guess_mime, resolve_sources

__all__ = ["resolve_sources", "guess_mime"]
