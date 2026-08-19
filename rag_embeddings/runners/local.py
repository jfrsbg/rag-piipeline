"""
The two backends that run on this machine: `docker://`, which shells out to the
Docker CLI, and `process://`, a child process for local development only.
Neither restarts anything; the queue redelivers instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from .base import Runner, TaskHandle, TaskResult, TaskSpec, TaskState

log = logging.getLogger(__name__)

# The label every container this dispatcher starts carries, so orphans left by
# a dispatcher that was killed can be found and removed:
#   docker rm -f $(docker ps -aq --filter label=rag.dispatch)
DISPATCH_LABEL = "rag.dispatch"

# Docker's own calls are local and fast; anything slower than this is the
# daemon being wedged, and blocking the dispatch loop on it helps nobody.
CLI_TIMEOUT = 60.0


class DockerRunner(Runner):
    """One container per document, on the local daemon."""

    backend = "docker"

    def __init__(
        self,
        *,
        volumes: Sequence[str] = (),
        network: str | None = None,
        pull: str | None = None,
        keep: bool = False,
        binary: str = "docker",
    ):
        self.volumes = list(volumes)
        self.network = network
        self.pull = pull
        # Exited containers are normally removed once their exit code has been
        # read. Keeping them is how you get `docker logs` on a failure after
        # the fact, at the cost of a growing `docker ps -a`.
        self.keep = keep
        self.binary = binary

    def __repr__(self) -> str:
        return f"DockerRunner(network={self.network!r}, volumes={len(self.volumes)})"

    # ----------------------------------------------------------- placement

    def launch(self, spec: TaskSpec) -> TaskHandle:
        result = self._cli(self.argv(spec))
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {spec.name}: {result.stderr.strip()}"
            )
        container_id = result.stdout.strip()
        return TaskHandle(
            id=container_id, name=spec.name, backend=self.backend,
            detail={"image": spec.image},
        )

    def status(self, handle: TaskHandle) -> TaskResult:
        result = self._cli([
            self.binary, "inspect",
            "--format", "{{.State.Status}} {{.State.ExitCode}}",
            handle.id,
        ])
        if result.returncode != 0:
            # The container is gone and we never read its exit code. Something
            # removed it out from under us; treating that as failed costs a
            # retry, and treating it as success would lose a document.
            return TaskResult(
                handle, TaskState.FAILED,
                reason=f"container not found: {result.stderr.strip()}",
            )

        status, _, code = result.stdout.strip().partition(" ")
        exit_code = int(code) if code.strip().lstrip("-").isdigit() else None

        if status in ("created", "restarting"):
            return TaskResult(handle, TaskState.PENDING)
        if status in ("running", "paused", "removing"):
            return TaskResult(handle, TaskState.RUNNING)
        if status in ("exited", "dead"):
            if exit_code == 0:
                return TaskResult(handle, TaskState.SUCCEEDED, exit_code=0)
            # 137 is SIGKILL, which here almost always means the OOM killer —
            # worth naming, because the fix is a bigger --memory and not a
            # retry, and the retry will fail identically. Any other code speaks
            # for itself and needs no reason beside it.
            reason = "killed (OOM?)" if exit_code == 137 else ""
            return TaskResult(handle, TaskState.FAILED, exit_code=exit_code, reason=reason)
        return TaskResult(handle, TaskState.UNKNOWN, reason=f"docker says {status!r}")

    def cancel(self, handle: TaskHandle) -> None:
        self._cli([self.binary, "kill", handle.id])

    def cleanup(self, handle: TaskHandle) -> None:
        if not self.keep:
            self._cli([self.binary, "rm", "--force", handle.id])

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        result = self._cli([self.binary, "logs", "--tail", str(tail), handle.id])
        if result.returncode != 0:
            return None
        # Docling writes progress to stderr, so a failure's explanation is
        # usually there rather than on stdout.
        return (result.stdout + result.stderr).strip() or None

    # --------------------------------------------------------------- private

    def argv(self, spec: TaskSpec) -> list[str]:
        """Build the `docker run` command line for `spec`."""
        argv = [self.binary, "run", "--detach", "--name", spec.name]
        # Not --rm: it deletes the container the instant it exits, and the exit
        # code is the only thing the dispatcher has to go on. `cleanup` removes
        # it once that code has been read.
        for key, value in spec.env.items():
            argv += ["--env", f"{key}={value}"]
        argv += ["--label", DISPATCH_LABEL]
        for key, value in spec.labels.items():
            argv += ["--label", f"{key}={value}"]
        # The machine's mounts, then the message's. The cache and the queue are
        # the same for every task this dispatcher starts; the document is not,
        # and it is mounted at the path the message named so that the uri the
        # worker is given is the uri it can open.
        for volume in (*self.volumes, *spec.mounts):
            argv += ["--volume", volume]
        if self.network:
            argv += ["--network", self.network]
        if self.pull:
            argv += ["--pull", self.pull]
        if spec.cpu:
            argv += ["--cpus", str(spec.cpu)]
        if spec.memory_mb:
            argv += ["--memory", f"{spec.memory_mb}m"]
        argv.append(spec.image)
        argv.extend(spec.command)
        return argv

    def _cli(self, argv: list[str]) -> subprocess.CompletedProcess:
        log.debug("%s", shlex.join(argv))
        try:
            return subprocess.run(
                argv, capture_output=True, text=True, timeout=CLI_TIMEOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.binary!r} is not on PATH — docker:// needs the CLI"
            ) from exc
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                argv, returncode=124, stdout="", stderr="docker CLI timed out",
            )


class ProcessRunner(Runner):
    """One child process per document, with output to a log file per task."""

    backend = "process"

    def __init__(self, *, log_dir: str | Path | None = None, cwd: str | None = None):
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.gettempdir()) / "rag-dispatch"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = cwd
        self._running: dict[str, tuple[subprocess.Popen, Path]] = {}

    def __repr__(self) -> str:
        return f"ProcessRunner(log_dir={str(self.log_dir)!r})"

    def launch(self, spec: TaskSpec) -> TaskHandle:
        if not spec.command:
            raise ValueError(
                "process:// has no image to take an entrypoint from — give it "
                "--task-command, e.g. --task-command 'python -m "
                "rag_embeddings.workers.parse_worker'"
            )
        # The dispatcher's default command is arguments to `python`, because
        # the image's entrypoint is `python`. There is no image here, so this
        # backend supplies that entrypoint itself — and it has to be *this*
        # interpreter, not whatever `python` resolves to on PATH.
        argv = list(spec.command)
        if argv[0].startswith("-"):
            argv = default_python_command() + argv
        # The child inherits this process's environment so that PATH, HOME and
        # the virtualenv still work; the spec's variables win where they
        # overlap, because those are the ones naming the document.
        env = {**os.environ, **spec.env}
        stream = self.log_dir / f"{spec.name}.log"
        handle_file = stream.open("wb")
        try:
            process = subprocess.Popen(
                argv, env=env, cwd=self.cwd,
                stdout=handle_file, stderr=subprocess.STDOUT,
            )
        finally:
            # Popen dups the descriptor; this process does not need its copy.
            handle_file.close()

        self._running[str(process.pid)] = (process, stream)
        return TaskHandle(
            id=str(process.pid), name=spec.name, backend=self.backend,
            detail={"log": str(stream)},
        )

    def status(self, handle: TaskHandle) -> TaskResult:
        entry = self._running.get(handle.id)
        if entry is None:
            return TaskResult(handle, TaskState.UNKNOWN, reason="not this runner's task")
        process, _stream = entry
        code = process.poll()
        if code is None:
            return TaskResult(handle, TaskState.RUNNING)
        if code == 0:
            return TaskResult(handle, TaskState.SUCCEEDED, exit_code=0)
        # A negative code is a signal number: -9 is the OOM killer or a `kill`,
        # and reporting it as "exit -9" would send someone hunting for a return
        # statement that does not exist.
        reason = f"killed by signal {-code}" if code < 0 else ""
        return TaskResult(handle, TaskState.FAILED, exit_code=code, reason=reason)

    def cancel(self, handle: TaskHandle) -> None:
        entry = self._running.get(handle.id)
        if entry is None:
            return
        process, _ = entry
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def cleanup(self, handle: TaskHandle) -> None:
        entry = self._running.pop(handle.id, None)
        if entry is not None:
            # Reap the zombie. The log file is left behind on purpose: it is
            # the only record of what the task printed.
            entry[0].poll()

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        path = Path(str(handle.detail.get("log", "")))
        if not path.exists():
            return None
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:]) or None

    def close(self) -> None:
        for handle_id in list(self._running):
            process, _ = self._running.pop(handle_id)
            if process.poll() is None:
                process.terminate()


def default_python_command() -> list[str]:
    """Return the interpreter running the dispatcher, not `python` on PATH."""
    return [sys.executable]
