#!/usr/bin/env python3
"""Query entrypoint: the search API. See rag_embeddings/api/app.py."""

import sys

from rag_embeddings.api.app import main

if __name__ == "__main__":
    sys.exit(main())
