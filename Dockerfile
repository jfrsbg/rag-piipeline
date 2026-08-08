# One image, two entrypoints. Both steps share the same dependency set, and
# the image is large enough (torch) that building it twice would cost more than
# the handful of MB step 1 doesn't use.
#
#   docker build -t rag-embeddings .
#   docker run --rm -v ./cache:/cache -v ./inbox:/data/inbox:ro \
#              rag-embeddings step1_parse.py /data/inbox
#   docker run --rm -v ./cache:/cache -v hf-models:/models \
#              -e RAG_DSN=postgresql://postgres:postgres@db/docs \
#              rag-embeddings step2_index.py
#
# GPU build: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

FROM python:3.12-slim

# opencv (via docling-ibm-models) needs these at import time, not just runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights live on a volume so a rebuild doesn't re-download 2 GB.
    HF_HOME=/models \
    # Both steps agree on the cache location; step 1 writes it, step 2 reads it.
    RAG_CACHE_DIR=/cache/parsed

WORKDIR /app

# Installed first and from the CPU index: the default PyPI torch wheel drags in
# the whole CUDA stack, which triples the image for no benefit on a laptop.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: bake the Docling layout models into the image instead of fetching
# them on first run. Build with --build-arg PREFETCH_MODELS=1.
ARG PREFETCH_MODELS=0
RUN if [ "$PREFETCH_MODELS" = "1" ]; then docling-tools models download; fi

COPY rag_embeddings ./rag_embeddings
COPY step1_parse.py step2_index.py ./

RUN mkdir -p /cache/parsed /models

# The step file is the argument, so neither step is the image's "default" one.
ENTRYPOINT ["python"]
CMD ["step2_index.py", "--help"]
