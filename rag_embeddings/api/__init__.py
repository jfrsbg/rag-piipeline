"""
The read side as an HTTP service; retrieval itself lives in `steps/search.py`.

Exports are resolved lazily so importing this package does not pull fastapi
and torch.
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
