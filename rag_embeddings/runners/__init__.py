"""
Runners, and the one function that chooses between them.

    from rag_embeddings.runners import open_runner
    runner = open_runner("docker://?volume=/cache:/cache&network=rag_default")

The url is the only place a backend is named — the same arrangement `queues`
uses, and for the same reason: the dispatcher takes a Runner and does not care
which one it got, so `RAG_RUNNER_URL=ecs://prod/parse-task?subnets=...` is the
whole migration from a laptop to a cluster.

    memory://                            record, launch nothing (--dry-run)
    process://                           a child process on this machine
    docker://                            a container on the local daemon
    ecs://<cluster>/<task-definition>    an ECS task
    k8s://<namespace>                    a Kubernetes Job

Everything after `?` configures placement: which volumes a local container
gets, which subnets a Fargate task lands in, which service account a Job runs
as. It lives in the url rather than in a flag apiece so that the whole
placement decision is one environment variable, and so adding a backend does
not add five flags that mean nothing to the other four.

An unknown option is an error rather than something quietly ignored: a typo in
`RAG_RUNNER_URL` would otherwise mean containers running with no volumes and
failing one document at a time.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from .base import (
    DEFAULT_STATUS_INTERVAL,
    DEFAULT_TASK_TIMEOUT,
    Runner,
    TaskHandle,
    TaskResult,
    TaskSpec,
    TaskState,
    task_name,
)
from .memory import RecordingRunner, reset as reset_memory, shared as shared_memory

DEFAULT_RUNNER_URL = "docker://"


class _Options:
    """The query string of a runner url, read once and checked for leftovers."""

    def __init__(self, query: str):
        self.raw = parse_qs(query)

    def one(self, key: str, default: str | None = None) -> str | None:
        values = self.raw.pop(key, None)
        return values[-1] if values else default

    def many(self, key: str) -> list[str]:
        """Repeated (`?volume=a&volume=b`) and comma-separated both work."""
        values = self.raw.pop(key, [])
        return [item for value in values for item in value.split(",") if item]

    def flag(self, key: str, default: bool = False) -> bool:
        value = self.one(key)
        return default if value is None else value.lower() in ("1", "true", "yes", "on")

    def number(self, key: str, default: int) -> int:
        value = self.one(key)
        return default if value is None else int(value)

    def pairs(self, key: str) -> dict[str, str]:
        """`?node_selector=disk=ssd,zone=a` -> {"disk": "ssd", "zone": "a"}"""
        out = {}
        for item in self.many(key):
            name, _, value = item.partition("=")
            out[name] = value
        return out

    def done(self, url: str) -> None:
        if self.raw:
            raise ValueError(
                f"unknown option(s) {sorted(self.raw)} in runner url {url!r}"
            )


def open_runner(url: str, **kwargs: Any) -> Runner:
    """Build the runner described by `url`.

    Keyword arguments win over the url, which is how a test injects a fake
    client and how a caller overrides one thing without rebuilding the string.

    A new backend is a Runner subclass implementing `launch` and `status` plus
    one branch here; the waiting, the timeout and the cancel-on-overrun it
    inherits from `base` are already the behaviour the dispatcher was tested
    against.
    """
    parsed = urlparse(str(url))
    scheme = parsed.scheme or "docker"
    options = _Options(parsed.query)

    if scheme == "memory":
        options.done(url)
        return shared_memory(parsed.netloc or "memory", **kwargs)

    if scheme == "process":
        from .local import ProcessRunner                       # noqa: PLC0415

        runner = ProcessRunner(
            log_dir=options.one("log_dir"),
            cwd=options.one("cwd"),
            **kwargs,
        )
        options.done(url)
        return runner

    if scheme == "docker":
        from .local import DockerRunner                        # noqa: PLC0415

        runner = DockerRunner(
            volumes=options.many("volume"),
            network=options.one("network"),
            pull=options.one("pull"),
            keep=options.flag("keep"),
            binary=options.one("binary", "docker"),
            **kwargs,
        )
        options.done(url)
        return runner

    if scheme == "ecs":
        from .ecs import EcsRunner                             # noqa: PLC0415

        runner = EcsRunner(
            cluster=parsed.netloc,
            task_definition=parsed.path.lstrip("/"),
            container=options.one("container", "parser"),
            launch_type=(options.one("launch_type", "FARGATE") or "").upper(),
            subnets=options.many("subnets"),
            security_groups=options.many("security_groups"),
            assign_public_ip=options.flag("assign_public_ip"),
            platform_version=options.one("platform_version"),
            region=options.one("region"),
            **kwargs,
        )
        options.done(url)
        return runner

    if scheme in ("k8s", "kubernetes"):
        from .kube import DEFAULT_TTL_SECONDS, KubernetesRunner  # noqa: PLC0415

        runner = KubernetesRunner(
            namespace=parsed.netloc or "default",
            service_account=options.one("service_account"),
            ttl_seconds=options.number("ttl", DEFAULT_TTL_SECONDS),
            image_pull_secret=options.one("image_pull_secret"),
            node_selector=options.pairs("node_selector"),
            **kwargs,
        )
        options.done(url)
        return runner

    raise NotImplementedError(
        f"no runner backend for {scheme!r}:// — subclass Runner and add a "
        f"branch to open_runner()"
    )


__all__ = [
    "Runner",
    "TaskSpec",
    "TaskHandle",
    "TaskResult",
    "TaskState",
    "RecordingRunner",
    "open_runner",
    "task_name",
    "shared_memory",
    "reset_memory",
    "DEFAULT_RUNNER_URL",
    "DEFAULT_TASK_TIMEOUT",
    "DEFAULT_STATUS_INTERVAL",
]
