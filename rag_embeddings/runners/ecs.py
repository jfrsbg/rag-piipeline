"""
ECS: one task per document, on Fargate or EC2 capacity. `TaskSpec.image` is
advisory — the image comes from the task definition; only the command and the
environment are overridden. boto3 is imported lazily.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .base import (
    Runner,
    TaskHandle,
    TaskResult,
    TaskSpec,
    TaskState,
    refuse_mounts,
)

log = logging.getLogger(__name__)

# lastStatus values, in the order ECS moves through them.
_PENDING = {"PROVISIONING", "PENDING", "ACTIVATING"}
_RUNNING = {"RUNNING", "DEACTIVATING", "STOPPING", "DEPROVISIONING"}


class EcsRunner(Runner):
    """`ecs://<cluster>/<task-definition>?subnets=...&security_groups=...`"""

    backend = "ecs"

    def __init__(
        self,
        cluster: str,
        task_definition: str,
        *,
        container: str = "parser",
        launch_type: str = "FARGATE",
        subnets: Sequence[str] = (),
        security_groups: Sequence[str] = (),
        assign_public_ip: bool = False,
        platform_version: str | None = None,
        region: str | None = None,
        client: Any = None,
    ):
        if not cluster:
            raise ValueError("ecs:// needs a cluster: ecs://my-cluster/parse-task")
        if not task_definition:
            raise ValueError(
                "ecs:// needs a task definition: ecs://my-cluster/parse-task"
            )
        if launch_type == "FARGATE" and not subnets:
            # Fargate has no default placement: RunTask without an
            # awsvpcConfiguration fails with an InvalidParameterException that
            # says nothing about subnets. Better to say it here.
            raise ValueError(
                "FARGATE needs at least one subnet: "
                "ecs://cluster/task-def?subnets=subnet-a,subnet-b"
            )
        self.cluster = cluster
        self.task_definition = task_definition
        self.container = container
        self.launch_type = launch_type
        self.subnets = list(subnets)
        self.security_groups = list(security_groups)
        self.assign_public_ip = assign_public_ip
        self.platform_version = platform_version
        self.region = region
        self._client = client

    def __repr__(self) -> str:
        return f"EcsRunner({self.cluster!r}, {self.task_definition!r})"

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3                                   # noqa: PLC0415

            self._client = boto3.client("ecs", region_name=self.region)
        return self._client

    # ----------------------------------------------------------- placement

    def launch(self, spec: TaskSpec) -> TaskHandle:
        refuse_mounts(spec, "ECS")
        response = self.client.run_task(**self.run_task_kwargs(spec))

        failures = response.get("failures") or []
        tasks = response.get("tasks") or []
        if failures or not tasks:
            # The commonest failure by far is capacity, and it is worth reading
            # the reason rather than only seeing "launch failed": the message
            # goes back on the queue either way, but a human deciding whether
            # to raise a service quota needs the string.
            raise RuntimeError(f"RunTask refused {spec.name}: {failures or 'no task'}")

        task = tasks[0]
        return TaskHandle(
            id=task["taskArn"], name=spec.name, backend=self.backend,
            detail={"cluster": self.cluster},
        )

    def status(self, handle: TaskHandle) -> TaskResult:
        response = self.client.describe_tasks(cluster=self.cluster, tasks=[handle.id])
        tasks = response.get("tasks") or []
        if not tasks:
            # ECS keeps stopped tasks for about an hour and then forgets them.
            # At a two-second poll that only happens if the dispatcher was
            # asleep, so it is not terminal — the task timeout will resolve it.
            return TaskResult(handle, TaskState.UNKNOWN, reason="task not described")

        task = tasks[0]
        last = task.get("lastStatus", "")
        if last in _PENDING:
            return TaskResult(handle, TaskState.PENDING)
        if last in _RUNNING:
            return TaskResult(handle, TaskState.RUNNING)

        container = next(
            (c for c in task.get("containers", []) if c.get("name") == self.container),
            None,
        ) or (task.get("containers") or [{}])[0]
        exit_code = container.get("exitCode")
        if exit_code == 0:
            return TaskResult(handle, TaskState.SUCCEEDED, exit_code=0)

        # A task that never started has no exit code at all, and the reason is
        # on the task rather than the container: an unpullable image, a subnet
        # with no route to ECR, a task role that cannot be assumed.
        reason = (
            container.get("reason")
            or task.get("stoppedReason")
            or task.get("stopCode")
            or "stopped without an exit code"
        )
        return TaskResult(handle, TaskState.FAILED, exit_code=exit_code, reason=reason)

    def cancel(self, handle: TaskHandle) -> None:
        try:
            self.client.stop_task(
                cluster=self.cluster, task=handle.id, reason="dispatcher timeout",
            )
        except Exception as exc:                            # noqa: BLE001
            log.warning("ecs: stop_task failed for %s: %s", handle.label, exc)

    def logs(self, handle: TaskHandle, tail: int = 50) -> str | None:
        """None: the task's logs are wherever its log driver put them."""
        return None

    # --------------------------------------------------------------- private

    def run_task_kwargs(self, spec: TaskSpec) -> dict[str, Any]:
        """Build the RunTask keyword arguments for `spec`."""
        override: dict[str, Any] = {
            "name": self.container,
            "environment": [
                {"name": key, "value": str(value)} for key, value in spec.env.items()
            ],
        }
        if spec.command:
            override["command"] = list(spec.command)

        overrides: dict[str, Any] = {"containerOverrides": [override]}
        if spec.cpu or spec.memory_mb:
            if self.launch_type == "FARGATE":
                # Fargate bills the task, not the container, and only accepts
                # sizing at task level — as strings, and only in the pairs it
                # supports. Container-level values here would be silently
                # capped by the task's.
                if spec.cpu:
                    overrides["cpu"] = str(int(spec.cpu * 1024))
                if spec.memory_mb:
                    overrides["memory"] = str(int(spec.memory_mb))
            else:
                if spec.cpu:
                    override["cpu"] = int(spec.cpu * 1024)
                if spec.memory_mb:
                    override["memory"] = int(spec.memory_mb)

        kwargs: dict[str, Any] = {
            "cluster": self.cluster,
            "taskDefinition": self.task_definition,
            "count": 1,
            "launchType": self.launch_type,
            "overrides": overrides,
            # Both are what turns `aws ecs list-tasks` into something you can
            # answer "which document is this?" from.
            "startedBy": "rag-dispatcher",
            "tags": [
                {"key": key, "value": str(value)}
                for key, value in {**spec.labels, "rag.task": spec.name}.items()
            ],
        }
        if self.subnets:
            kwargs["networkConfiguration"] = {
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "securityGroups": self.security_groups,
                    "assignPublicIp": "ENABLED" if self.assign_public_ip else "DISABLED",
                }
            }
        if self.platform_version:
            kwargs["platformVersion"] = self.platform_version
        return kwargs
