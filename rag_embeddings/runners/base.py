"""
The runner seam: a backend supplies placement, `base` supplies waiting,
timeouts and verdicts. Backends never retry — a failed task is reported failed
and the queue redelivers.
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
    """Where a task is, in one vocabulary for every backend."""

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
    """What to run: `cpu` in cores, `memory_mb` in MiB, converted per backend."""

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
    """Build a unique name from a uri that Docker, ECS and Kubernetes accept."""
    stem = _UNSAFE.sub("-", hint.lower().rsplit("/", 1)[-1]).strip("-")
    suffix = uuid.uuid4().hex[:8]
    room = 63 - len(prefix) - len(suffix) - 2
    return f"{prefix}-{stem[:room].strip('-') or 'doc'}-{suffix}"


def refuse_mounts(spec: TaskSpec, backend: str) -> None:
    """Raise if a spec needs a host mount the backend cannot give it."""
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
        """Start one task and return once it is accepted, not once it is done."""

    @abstractmethod
    def status(self, handle: TaskHandle) -> TaskResult:
        """Report where that task is now; cheap enough for a poll loop."""

    def cancel(self, handle: TaskHandle) -> None:
        """Kill the task, best-effort and idempotent."""
        log.warning("%s: cannot cancel %s", self.backend, handle.label)

    def cleanup(self, handle: TaskHandle) -> None:
        """Release what a finished task still holds; call it after `status`."""

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        """Return the task's last lines, or None if the backend has none."""
        return None

    def close(self) -> None:
        """Release whatever the backend holds."""

    # ---------------------------------------------------------------- above

    def wait(
        self,
        handle: TaskHandle,
        *,
        timeout: float | None = DEFAULT_TASK_TIMEOUT,
        interval: float = DEFAULT_STATUS_INTERVAL,
        should_stop: Callable[[], bool] | None = None,
    ) -> TaskResult:
        """Poll `status` until the task is terminal, cancelling on timeout."""
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
        """Launch, wait and clean up, synchronously."""
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
