#!/usr/bin/env python3
"""Query entrypoint: semantic search. See rag_embeddings/steps/search.py."""

import sys

from rag_embeddings.steps.search import main

if __name__ == "__main__":
    sys.exit(main())
