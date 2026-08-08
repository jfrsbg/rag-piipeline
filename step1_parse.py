#!/usr/bin/env python3
"""Step 1 entrypoint: parse and cache. See rag_embeddings/steps/parse.py."""

import sys

from rag_embeddings.steps.parse import main

if __name__ == "__main__":
    sys.exit(main())
