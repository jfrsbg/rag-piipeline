#!/usr/bin/env python3
"""Index worker entrypoint. See rag_embeddings/workers/index_worker.py."""

import sys

from rag_embeddings.workers.index_worker import main

if __name__ == "__main__":
    sys.exit(main())
