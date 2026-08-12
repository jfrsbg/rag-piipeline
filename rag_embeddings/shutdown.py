"""
Stopping a service without losing the document it was working on.

A batch job could be killed at any point and simply re-run. A service cannot:
`docker compose stop`, a rolling update and a scale-down all arrive as SIGTERM
in the middle of whatever the container happens to be doing, and Python's
default handler ends the process there — mid-parse, with a message still
claimed. Nothing is lost (the claim expires and the message is redelivered),
but every deploy then pays a visibility timeout before that document is retried.

So the signal sets a flag instead. The consume loop checks it between messages,
finishes the one in flight, acks it, and returns — which is a clean exit, not a
crash, and so does not trip the restart policy.

The second signal is deliberately not swallowed: if a document is genuinely
stuck, an operator pressing Ctrl-C twice means it, and waiting for a grace
period they have already given up on helps nobody.
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
    """Yield a predicate that becomes true once a stop signal arrives.

    Restores the previous handlers on the way out, so importing this into a
    test or a notebook does not permanently change how the process dies.
    """
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
