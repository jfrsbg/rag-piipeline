"""
A runner that starts nothing and records everything.

This is what `--dry-run` uses and what the dispatcher's tests run against. It
is the runner equivalent of `queues/memory.py`: the simplest thing that
satisfies the contract in `base`, so a failure here is a bug in the test rather
than in the backend.

It is also the honest way to answer "what would this actually launch?" — the
argv, the environment and the resources are built by the dispatcher, not by the
backend, so a dry run exercises every line that decides them.
"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, Callable

from .base import Runner, TaskHandle, TaskResult, TaskSpec, TaskState

log = logging.getLogger(__name__)


class RecordingRunner(Runner):
    """Records launches; reports whatever `outcome` says.

    `outcome` is consulted at status time rather than at launch, so a test can
    change its mind about a task that is already running — which is how the
    "container died halfway" case gets covered without a container.

    `polls_before_done` makes a task take a measurable amount of time without
    any sleeping: it stays RUNNING for that many status calls. Real elapsed
    time in a test is a flake waiting to happen.
    """

    backend = "memory"

    def __init__(
        self,
        name: str = "memory",
        *,
        outcome: Callable[[TaskSpec], TaskState] | None = None,
        on_launch: Callable[[TaskSpec], None] | None = None,
        polls_before_done: int = 0,
    ):
        self.name = name
        self.outcome = outcome or (lambda spec: TaskState.SUCCEEDED)
        self.on_launch = on_launch
        self.polls_before_done = polls_before_done
        self.launched: list[TaskSpec] = []
        self.cancelled: list[str] = []
        self.cleaned: list[str] = []
        self._specs: dict[str, TaskSpec] = {}
        self._polls: dict[str, int] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"RecordingRunner({self.name!r}, launched={len(self.launched)})"

    @property
    def running(self) -> int:
        """Tasks launched and not yet cleaned up — the concurrency assertion."""
        with self._lock:
            return len(self._specs) - len(self.cleaned)

    def launch(self, spec: TaskSpec) -> TaskHandle:
        if self.on_launch is not None:
            # May raise: that is how a test covers "the orchestrator refused".
            self.on_launch(spec)
        with self._lock:
            task_id = f"{self.name}-{next(self._ids)}"
            self.launched.append(spec)
            self._specs[task_id] = spec
            self._polls[task_id] = 0
        log.debug("would run: %s %s", spec.image, " ".join(spec.command))
        return TaskHandle(id=task_id, name=spec.name, backend=self.backend)

    def status(self, handle: TaskHandle) -> TaskResult:
        with self._lock:
            spec = self._specs.get(handle.id)
            if spec is None:
                return TaskResult(handle, TaskState.UNKNOWN, reason="no such task")
            self._polls[handle.id] += 1
            polls = self._polls[handle.id]
        if polls <= self.polls_before_done:
            return TaskResult(handle, TaskState.RUNNING)
        state = self.outcome(spec)
        code = 0 if state is TaskState.SUCCEEDED else 1
        return TaskResult(
            handle, state,
            exit_code=code if state.done else None,
            reason="" if state is TaskState.SUCCEEDED else "recorded failure",
        )

    def cancel(self, handle: TaskHandle) -> None:
        self.cancelled.append(handle.id)

    def cleanup(self, handle: TaskHandle) -> None:
        self.cleaned.append(handle.id)

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        return None

    # ------------------------------------------------------------ inspection

    def uris(self) -> list[str]:
        """The document each launched task was told to parse."""
        return [spec.env.get("RAG_DOC_URI", "") for spec in self.launched]

    def reset(self) -> None:
        with self._lock:
            self.launched.clear()
            self.cancelled.clear()
            self.cleaned.clear()
            self._specs.clear()
            self._polls.clear()


_REGISTRY: dict[str, RecordingRunner] = {}
_REGISTRY_LOCK = threading.Lock()


def shared(name: str = "memory", **kwargs: Any) -> RecordingRunner:
    """The one recording runner with this name in this interpreter.

    `open_runner("memory://")` has to return the same object every call or a
    caller could not inspect what its dispatcher launched — the same reason
    `memory://` queues live in a registry.
    """
    with _REGISTRY_LOCK:
        runner = _REGISTRY.get(name)
        if runner is None:
            runner = _REGISTRY[name] = RecordingRunner(name, **kwargs)
        return runner


def reset() -> None:
    """Drop every shared runner. Test isolation, nothing else."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
