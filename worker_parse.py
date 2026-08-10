#!/usr/bin/env python3
"""Parse worker entrypoint. See rag_embeddings/workers/parse_worker.py."""

import sys

from rag_embeddings.workers.parse_worker import main

if __name__ == "__main__":
    sys.exit(main())
