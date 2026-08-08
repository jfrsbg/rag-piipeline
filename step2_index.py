#!/usr/bin/env python3
"""Step 2 entrypoint: extract and store. See rag_embeddings/steps/index.py."""

import sys

from rag_embeddings.steps.index import main

if __name__ == "__main__":
    sys.exit(main())
