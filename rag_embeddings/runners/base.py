"""
The runner seam.

The dispatcher hands one document to a runner and never to an orchestrator.
A backend supplies placement — start a container somewhere, tell me whether it
is still going, kill it if it overruns — and everything built on top of that
lives here, once: waiting for a task, timing it out, turning an exit code into
a verdict.

That split is the same one `queues` makes, for the same reason. Moving from
`docker run` on a laptop to `RunTask` on Fargate must not be an opportunity to
get the retry semantics subtly different, so no backend gets to decide them:
retries belong to the queue, which already has them.

The contract a backend has to honour:

  * `launch` returns as soon as the task is *accepted*, never when it finishes —
    a dispatcher that blocks on a parse can only ever run one at a time;
  * a handle returned by `launch` stays valid for `status` until `cleanup`;
  * `status` reports a terminal state exactly once the task can no longer
    change it, and SUCCEEDED only for a zero exit;
  * `cancel` is best-effort and idempotent: cancelling a task that already
    exited is not an error.

Nothing here retries. A task that fails is reported as failed, the dispatcher
nacks the message that produced it, and the queue redelivers — which is the
path that was already tested. Backends that offer their own retry (k8s
`backoffLimit`, ECS service scheduling) have it turned off on purpose: two
layers of retry means a poison document is attempted `max_attempts × backoff`
times and dead-letters much later than the operator expects.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

log = logging.getLogger(__name__)

# A parse is minutes, not hours. The cap exists so a wedged container is
# eventually killed and its document retried, rather than holding a slot in the
# dispatcher forever.
DEFAULT_TASK_TIMEOUT = 900.0
# How often a running task is asked how it is doing. Every backend charges for
# this — an API call on ECS and k8s, a fork on Docker — so it is deliberately
# slower than the queue's poll interval.
DEFAULT_STATUS_INTERVAL = 2.0


class TaskState(str, Enum):
    """Where a task is. Four backends, one vocabulary."""

    PENDING = "pending"        # accepted, not started: pulling, scheduling
    RUNNING = "running"
    SUCCEEDED = "succeeded"    # exited zero
    FAILED = "failed"          # exited non-zero, was killed, or never started
    UNKNOWN = "unknown"        # the backend would not say; not terminal

    @property
    def done(self) -> bool:
        return self in (TaskState.SUCCEEDED, TaskState.FAILED)


@dataclass(frozen=True)
class TaskSpec:
    """What to run. Deliberately the intersection of what all three can do.

    Anything a single backend supports and the others do not — placement
    constraints, node selectors, capacity providers — is configured on the
    runner (and therefore in its url), not here: a spec that only one backend
    understands is a spec the dispatcher cannot hand to another one.

    `cpu` is in cores and `memory_mb` in MiB because those are the units every
    backend can be told in; each converts on the way out.
    """

    name: str
    image: str
    command: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cpu: float | None = None
    memory_mb: int | None = None
    # Carried onto the task as labels/tags so a container can be traced back to
    # the message that caused it without reading its logs.
    labels: Mapping[str, str] = field(default_factory=dict)
    # The document, as `host:container:ro` — the one piece of placement that
    # comes from the message rather than from the machine, which is why it is
    # here and the runner's own `volumes` are not. A task is told to parse a
    # path and the path has to exist for it; on one machine that means a bind
    # mount of that file, and it is per-document because each task gets its own
    # message. Empty whenever the uri is remote, because then there is nothing
    # local to mount and the worker fetches the bytes itself.
    #
    # Only the local backends can honour it: ECS and Kubernetes have no host to
    # mount from, so a document that needs this cannot be dispatched there —
    # they say so at launch rather than starting a task that cannot find it.
    mounts: Sequence[str] = ()


@dataclass(frozen=True)
class TaskHandle:
    """The backend's handle on a running task, opaque above it."""

    id: str
    name: str
    backend: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short enough for a log line, specific enough to grep."""
        return f"{self.name} [{self.id[:12]}]"


@dataclass(frozen=True)
class TaskResult:
    handle: TaskHandle
    state: TaskState
    exit_code: int | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.state is TaskState.SUCCEEDED

    def describe(self) -> str:
        parts = [self.state.value]
        if self.exit_code is not None:
            parts.append(f"exit {self.exit_code}")
        if self.reason:
            parts.append(self.reason)
        return ", ".join(parts)


_UNSAFE = re.compile(r"[^a-z0-9-]+")


def task_name(hint: str, prefix: str = "parse") -> str:
    """A name that Docker, ECS and Kubernetes all accept, from a uri.

    Kubernetes is the strict one: lowercase alphanumerics and dashes, 63
    characters, and it must not collide with a job that already exists. The
    uuid suffix is what makes a redelivered document a second task rather than
    a name conflict, and the readable stem is what makes `docker ps` and
    `kubectl get jobs` worth looking at.
    """
    stem = _UNSAFE.sub("-", hint.lower().rsplit("/", 1)[-1]).strip("-")
    suffix = uuid.uuid4().hex[:8]
    room = 63 - len(prefix) - len(suffix) - 2
    return f"{prefix}-{stem[:room].strip('-') or 'doc'}-{suffix}"


def refuse_mounts(spec: TaskSpec, backend: str) -> None:
    """Raise if a spec needs a host mount the backend cannot give it.

    For the backends with no host to mount from. A document that only exists on
    the dispatcher's disk cannot be parsed by a task in a cluster, and the
    honest failure is here, at launch, rather than a container that starts fine
    and reports "nothing to parse" — the fix is to put the document somewhere
    the task can reach, which is a different day's work from a retry.
    """
    if spec.mounts:
        raise RuntimeError(
            f"{spec.name} needs {list(spec.mounts)} mounted, and {backend} has "
            f"no host to mount from — the document must be somewhere the task "
            f"can fetch it (s3://...), not a path on the dispatcher"
        )


class Runner(ABC):
    """Somewhere a container can run. Subclasses implement placement only."""

    backend: str = "runner"

    # ----------------------------------------------------------- placement

    @abstractmethod
    def launch(self, spec: TaskSpec) -> TaskHandle:
        """Start one task and return once it is accepted, not once it is done.

        Raising means the task was never started, which the dispatcher treats
        as a failure of the message rather than of the document.
        """

    @abstractmethod
    def status(self, handle: TaskHandle) -> TaskResult:
        """Where that task is now. Cheap enough to call on a poll loop."""

    def cancel(self, handle: TaskHandle) -> None:
        """Best-effort kill. Called on timeout and on shutdown."""
        log.warning("%s: cannot cancel %s", self.backend, handle.label)

    def cleanup(self, handle: TaskHandle) -> None:
        """Release what a *finished* task still holds. Default: nothing.

        Separate from `cancel` because the thing being released is usually the
        record `status` reads — deleting it any earlier loses the exit code.
        """

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        """The task's last lines, if the backend can produce them cheaply.

        Only ever used to put a failure in the dispatcher's log next to the
        document that caused it. `None` means "look somewhere else", which for
        a cluster backend is the honest answer.
        """
        return None

    def close(self) -> None:
        """Release whatever the backend holds. Default: nothing to release."""

    # ---------------------------------------------------------------- above

    def wait(
        self,
        handle: TaskHandle,
        *,
        timeout: float | None = DEFAULT_TASK_TIMEOUT,
        interval: float = DEFAULT_STATUS_INTERVAL,
        should_stop: Callable[[], bool] | None = None,
    ) -> TaskResult:
        """Poll `status` until the task is terminal.

        For a single task — `run` below, a test, a REPL. The dispatcher does
        not use this: it multiplexes many tasks over one loop and cannot afford
        to block on any of them.

        A task that overruns `timeout` is cancelled, not merely abandoned: an
        abandoned container keeps its memory and its CPU while the document it
        was parsing is retried on another one.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.status(handle)
            if result.state.done:
                return result
            if should_stop is not None and should_stop():
                return TaskResult(handle, TaskState.UNKNOWN, reason="stopped waiting")
            if deadline is not None and time.monotonic() >= deadline:
                log.error("%s: %s overran %.0fs, cancelling", self.backend,
                          handle.label, timeout)
                self.cancel(handle)
                return TaskResult(
                    handle, TaskState.FAILED, reason=f"timed out after {timeout:.0f}s"
                )
            time.sleep(interval)

    def run(self, spec: TaskSpec, **kwargs: Any) -> TaskResult:
        """Launch, wait, clean up. The synchronous convenience."""
        handle = self.launch(spec)
        try:
            return self.wait(handle, **kwargs)
        finally:
            self.cleanup(handle)

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
