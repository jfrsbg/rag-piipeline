# syntax=docker/dockerfile:1
#
# The pipeline image: the producer and the two queue services. One image,
# several entrypoints — they share the same dependency set, and the image is
# large enough (torch) that building it twice would cost more than the handful
# of MB the parse service doesn't use.
#
# The read side is not here. It is a service with an HTTP stack of its own, so
# it has its own image: see Dockerfile.api.
#
#   docker build -t rag-embeddings .
#   # publish work, then leave the services up to consume it
#   docker run --rm -v queue:/queue -v ./inbox:/data/inbox:ro \
#              rag-embeddings enqueue.py files /data/inbox
#   docker run -d -v ./cache:/cache -v queue:/queue -v ./inbox:/data/inbox:ro \
#              rag-embeddings worker_parse.py
#   docker run -d -v ./cache:/cache -v queue:/queue -v hf-models:/models \
#              -e RAG_DSN=postgresql://postgres:postgres@db/docs \
#              rag-embeddings worker_index.py
#
# GPU build: --build-arg TORCH_EXTRA=cu126

FROM python:3.12-slim

# Pinned to the version that wrote uv.lock: `uv sync --locked` refuses a lock
# whose format revision it does not recognise, which is the behaviour you want
# from a reproducible build rather than a surprise re-resolve.
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

# opencv (via docling-ibm-models) needs these at import time, not just runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Only the settings that uv itself reads live above the install layers.
# Everything tunable is set further down, so editing a runtime knob does not
# invalidate the ~1.7 GB dependency layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The uv cache is a mount, not a layer, so hardlinking out of it into the
    # venv would cross filesystems and warn on every package.
    UV_LINK_MODE=copy \
    # Containers are one-shot (`run --rm`), so without this every run pays the
    # import-time compile of torch and docling again. Costs ~200 MB of .pyc.
    UV_COMPILE_BYTECODE=1 \
    # `uv run` is not used at runtime; the venv is on PATH directly so that
    # ENTRYPOINT ["python"] resolves to it.
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Which build of torch to install. The extras are declared conflicting in
# pyproject.toml, so exactly one may be selected; see the comment there for why
# torch is named at all when sentence-transformers already depends on it.
ARG TORCH_EXTRA=cpu

# Dependencies resolve from the lock alone, so this layer is keyed on
# pyproject.toml + uv.lock and survives every source edit. The wheels behind it
# are ~2 GB to download, and the cache mount is what keeps a rebuild from
# paying that again: it is not part of the image, it survives layer
# invalidation, and it is shared by every build on this machine.
# `docker builder prune` clears it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra ${TORCH_EXTRA}

# Model weights live on a volume so a rebuild doesn't re-download 2.6 GB.
ENV HF_HOME=/models \
    # Both steps agree on the cache location; step 1 writes it, step 2 reads it.
    RAG_CACHE_DIR=/cache/parsed \
    # Docling runs its layout model through torch.compile by default, and the
    # inductor CPU backend shells out to g++ to build kernels. This image has no
    # toolchain, so every layout batch died with InvalidCxxCompiler and the PDF
    # came out empty. Eager mode needs no compiler; the alternative is to
    # apt-get install g++ (~250 MB) and pay a cold compile on the first batch.
    DOCLING_INFERENCE_COMPILE_TORCH_MODELS=0

# Optional: bake the weights into the image instead of fetching them on first
# run — the Docling layout models (~0.5 GB) and the embedding model (~2.1 GB).
# Build with --build-arg PREFETCH_MODELS=1. This only pays off where the volume
# cannot: CI, an air-gapped host, or an image shipped to someone else. Locally
# the volume already holds them, and a baked /models is shadowed by any mount
# over it except the first-time initialisation of an empty named volume.
ARG PREFETCH_MODELS=0
ARG PREFETCH_EMBED_MODEL=BAAI/bge-m3
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
      docling-tools models download && \
      python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" \
             "$PREFETCH_EMBED_MODEL"; \
    fi

COPY rag_embeddings ./rag_embeddings
# The two queue consumers, plus the producer that feeds them.
COPY worker_parse.py worker_index.py enqueue.py ./

# Installs the project itself against the deps already present above. Also puts
# the `rag-parse-worker` / `rag-index-worker` / `rag-enqueue` console scripts on
# PATH, which the files above shadow rather than replace.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ${TORCH_EXTRA}

RUN mkdir -p /cache/parsed /models /queue

# Both consumers block on an empty queue and never exit on their own, so a
# container started from this image is expected to be long-lived. The entrypoint
# file is the argument: neither service is the image's "default" one, and the
# help text is the only thing safe to run without knowing which was meant.
ENTRYPOINT ["python"]
CMD ["worker_parse.py", "--help"]
