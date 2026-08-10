#!/usr/bin/env python3
"""Producer entrypoint. See rag_embeddings/workers/enqueue.py."""

import sys

from rag_embeddings.workers.enqueue import main

if __name__ == "__main__":
    sys.exit(main())
