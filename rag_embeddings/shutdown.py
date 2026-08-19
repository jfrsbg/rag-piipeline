"""Graceful shutdown: a stop signal sets a flag the consume loop checks between
messages, so the document in flight is finished and acked. A second signal is
not swallowed — it kills the process as usual.
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from typing import Callable, Iterator

log = logging.getLogger(__name__)

# The signals an orchestrator uses to ask for a shutdown. SIGTERM is what
# Docker and Kubernetes send; SIGINT is Ctrl-C in a terminal.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@contextmanager
def stop_requested() -> Iterator[Callable[[], bool]]:
    """Yield a predicate that becomes true once a stop signal arrives."""
    flagged = False

    def requested() -> bool:
        return flagged

    def handle(signum, _frame):
        nonlocal flagged
        if flagged:
            # Already asked once and still here: restore the default and
            # re-raise so the second signal does what the operator expects.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        flagged = True
        log.info(
            "%s received: finishing the message in flight, then stopping",
            signal.Signals(signum).name,
        )

    previous = {}
    for sig in STOP_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handle)
        except ValueError:
            # Not the main thread — the caller is embedding this rather than
            # running it as a service, and owns its own lifecycle.
            log.debug("cannot install a handler for %s here", sig)

    try:
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
