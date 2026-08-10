"""
The two steps as queue consumers, plus the producer that feeds them.

    enqueue.files  ->  to-parse  ->  parse_worker  ->  to-index  ->  index_worker

`steps.parse` and `steps.index` still do the same work in batch — they are the
right thing for a backfill, a laptop and the wiring test. These are the same
work reshaped for a pool: one document per message, nothing enumerated, nothing
expensive built inside the loop.
"""

from .enqueue import enqueue_cached, enqueue_files, enqueue_stale
from .index_worker import run as run_index_worker
from .parse_worker import run as run_parse_worker

__all__ = [
    "enqueue_files",
    "enqueue_cached",
    "enqueue_stale",
    "run_parse_worker",
    "run_index_worker",
]
