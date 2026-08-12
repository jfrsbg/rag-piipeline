"""
The read side as an HTTP service.

`steps/search.py` still does the retrieval — embed the query, rank the chunks,
flag anything indexed under another profile. This package is only the part
that turns it into something long-lived: process-wide resources in `app.py`,
the wire contract in `schemas.py`, one route in `routes.py`.

Imported lazily for the same reason the top-level package defers its exports:
`import rag_embeddings.api` pulls fastapi and, through the embedder, torch.
Nothing that is not the service should pay for that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["create_app", "app"]


def __getattr__(name: str):
    if name in __all__:
        from . import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:                               # pragma: no cover
    from .app import app, create_app
