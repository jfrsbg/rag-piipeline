"""
FastAPI app for the query path.

The embedder, pool and profile check are built once in the lifespan.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from ..cli import configure_logging
from ..config import ApiSettings, Settings
from ..embedder import Embedder
from ..storage.connection import pool as connection_pool
from ..steps.search import warn_on_profile_mismatch
from .routes import router

log = logging.getLogger(__name__)


@dataclass
class Resources:
    """Process-wide resources shared by every request."""

    settings: Settings
    pool: object
    embedder: Embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    api = ApiSettings.from_env()
    # Here rather than in main() so it also applies under
    # `uvicorn rag_embeddings.api.app:app`. uvicorn configures its own loggers
    # and leaves the root one alone, which would send this package's warnings —
    # the profile mismatch above all — to logging's bare fallback handler.
    configure_logging(api.log_level)

    settings = Settings.from_env()

    pool = connection_pool(settings.dsn, max_size=api.pool_size)
    # Fail here rather than on the first request: a container that cannot
    # reach the database should not report itself as serving.
    pool.open(wait=True, timeout=30)

    log.info("loading %s", settings.profile.model_id)
    embedder = Embedder(
        settings.profile, token_budget=settings.embed_token_budget
    )

    with pool.connection() as conn:
        warn_on_profile_mismatch(conn, embedder)

    app.state.resources = Resources(settings, pool, embedder)
    log.info("ready: profile %s", settings.profile.version)
    try:
        yield
    finally:
        pool.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="rag-embeddings search",
        version="0.1.0",
        description=(
            "The read side of the pipeline: embed a query with the same "
            "profile the corpus was indexed under, rank chunks by cosine "
            "distance."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


# Module-level so `uvicorn rag_embeddings.api.app:app` works. Nothing
# expensive happens at import; the model loads in the lifespan above.
app = create_app()


def main() -> int:
    """Entrypoint behind `rag-serve` and serve.py."""
    import uvicorn

    api = ApiSettings.from_env()
    # The application's own logging is set up in the lifespan, so that it is
    # configured the same way whether uvicorn was started from here or by name.
    uvicorn.run(app, host=api.host, port=api.port)
    return 0
