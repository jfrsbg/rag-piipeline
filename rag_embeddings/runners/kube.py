"""
Kubernetes: one Job per document.

A Job rather than a bare Pod, because a Pod that is evicted — a spot node going
away, a drained node — is simply gone, and nothing would ever tell the
dispatcher. A Job with `backoffLimit: 0` is the narrowest thing that still
records a terminal outcome: exactly one Pod, one attempt, and a status the
dispatcher can read after the fact.

`backoffLimit: 0` is the important line. Kubernetes will happily retry a failed
Pod six times with a backoff, and if it did, a poison document would be
attempted six times per delivery and three deliveries later would finally
dead-letter — twenty minutes after the operator expected it. Retries belong to
the queue, which already counts them.

The client is imported inside `api`, for the same reason boto3 is in `ecs.py`.
The Job is built as a plain dict rather than through the V1* model classes:
it is the same JSON either way, and a dict is one import instead of eight and
is readable in a test assertion.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    Runner,
    TaskHandle,
    TaskResult,
    TaskSpec,
    TaskState,
    refuse_mounts,
)

log = logging.getLogger(__name__)

# How long a finished Job sticks around before the cluster reaps it. Long
# enough that `kubectl get jobs` after a failure still shows something, short
# enough that a day of documents does not fill etcd with completed Jobs.
DEFAULT_TTL_SECONDS = 600


class KubernetesRunner(Runner):
    """`k8s://<namespace>?service_account=...&ttl=600`"""

    backend = "k8s"

    def __init__(
        self,
        namespace: str = "default",
        *,
        service_account: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        image_pull_secret: str | None = None,
        node_selector: dict[str, str] | None = None,
        api: Any = None,
    ):
        self.namespace = namespace or "default"
        self.service_account = service_account
        self.ttl_seconds = ttl_seconds
        self.image_pull_secret = image_pull_secret
        self.node_selector = node_selector or {}
        self._api = api

    def __repr__(self) -> str:
        return f"KubernetesRunner({self.namespace!r})"

    @property
    def api(self) -> Any:
        if self._api is None:
            from kubernetes import client, config                # noqa: PLC0415

            try:
                # In a cluster this reads the projected service account token;
                # it is tried first so that a developer's stale kubeconfig can
                # never win over the pod's own identity.
                config.load_incluster_config()
            except Exception:                                    # noqa: BLE001
                config.load_kube_config()
            self._api = client.BatchV1Api()
        return self._api

    # ----------------------------------------------------------- placement

    def launch(self, spec: TaskSpec) -> TaskHandle:
        refuse_mounts(spec, "Kubernetes")
        body = self.job_body(spec)
        job = self.api.create_namespaced_job(namespace=self.namespace, body=body)
        name = _get(job, "metadata", "name") or spec.name
        return TaskHandle(
            id=name, name=spec.name, backend=self.backend,
            detail={"namespace": self.namespace},
        )

    def status(self, handle: TaskHandle) -> TaskResult:
        try:
            job = self.api.read_namespaced_job_status(
                name=handle.id, namespace=self.namespace,
            )
        except Exception as exc:                                 # noqa: BLE001
            if _is_not_found(exc):
                # The TTL collected it before we read it. That only happens if
                # the dispatcher stopped polling for longer than the TTL, and
                # the outcome is genuinely unrecoverable rather than failed.
                return TaskResult(handle, TaskState.UNKNOWN, reason="job already reaped")
            raise

        succeeded = _get(job, "status", "succeeded") or 0
        failed = _get(job, "status", "failed") or 0
        active = _get(job, "status", "active") or 0

        if succeeded:
            return TaskResult(handle, TaskState.SUCCEEDED, exit_code=0)
        if failed:
            return TaskResult(handle, TaskState.FAILED, reason=_failure_reason(job))
        if active:
            return TaskResult(handle, TaskState.RUNNING)
        # Created, not scheduled: no node with room yet, or the image is still
        # being pulled.
        return TaskResult(handle, TaskState.PENDING)

    def cancel(self, handle: TaskHandle) -> None:
        try:
            # Background propagation, or the Job object is deleted and its Pod
            # is left running with nothing pointing at it.
            self.api.delete_namespaced_job(
                name=handle.id, namespace=self.namespace,
                propagation_policy="Background",
            )
        except Exception as exc:                                 # noqa: BLE001
            if not _is_not_found(exc):
                log.warning("k8s: could not delete job %s: %s", handle.id, exc)

    def cleanup(self, handle: TaskHandle) -> None:
        """Nothing: `ttlSecondsAfterFinished` is the cluster doing this for us.

        Deleting the Job here would work too, and would also delete the Pod
        whose logs are the only explanation of a failure, at the exact moment
        someone wants them.
        """

    # --------------------------------------------------------------- private

    def job_body(self, spec: TaskSpec) -> dict[str, Any]:
        """The Job manifest, built where a test can assert on it."""
        container: dict[str, Any] = {
            "name": "parser",
            "image": spec.image,
            "env": [
                {"name": key, "value": str(value)} for key, value in spec.env.items()
            ],
        }
        if spec.command:
            # `args`, not `command`: overriding `command` replaces the image's
            # entrypoint, which is how you get "exec: python: not found" from a
            # container whose entrypoint was a wrapper script.
            container["args"] = list(spec.command)
        if spec.cpu or spec.memory_mb:
            resources = {}
            if spec.cpu:
                resources["cpu"] = f"{int(spec.cpu * 1000)}m"
            if spec.memory_mb:
                resources["memory"] = f"{int(spec.memory_mb)}Mi"
            # Requests and limits the same: the parser's memory is what decides
            # whether the node OOMs, and a burstable Pod is the one the kubelet
            # evicts first under pressure.
            container["resources"] = {"requests": dict(resources), "limits": dict(resources)}

        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
        }
        if self.service_account:
            pod_spec["serviceAccountName"] = self.service_account
        if self.image_pull_secret:
            pod_spec["imagePullSecrets"] = [{"name": self.image_pull_secret}]
        if self.node_selector:
            pod_spec["nodeSelector"] = dict(self.node_selector)

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": spec.name,
                "labels": {"app": "rag-parser", **_label_safe(spec.labels)},
            },
            "spec": {
                # One Pod, one attempt. See the module docstring.
                "backoffLimit": 0,
                "completions": 1,
                "parallelism": 1,
                "ttlSecondsAfterFinished": self.ttl_seconds,
                "template": {
                    "metadata": {"labels": {"app": "rag-parser"}},
                    "spec": pod_spec,
                },
            },
        }


def _get(obj: Any, *path: str) -> Any:
    """Read a field from either a V1* model or the dict a fake returns."""
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "status", None) == 404


def _failure_reason(job: Any) -> str:
    conditions = _get(job, "status", "conditions") or []
    for condition in conditions:
        if _get(condition, "type") == "Failed":
            return _get(condition, "message") or _get(condition, "reason") or "job failed"
    return "job failed"


def _label_safe(labels: Any) -> dict[str, str]:
    """Kubernetes label values are 63 chars of [-A-Za-z0-9_.]; a uri is not.

    Dropping what does not fit is better than failing the Job creation: these
    are for finding a Pod by eye, and a document whose uri is too long still
    has to be parsed.
    """
    out = {}
    for key, value in dict(labels or {}).items():
        text = str(value)
        if len(text) <= 63 and all(c.isalnum() or c in "-_." for c in text):
            out[key] = text
    return out
