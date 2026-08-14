"""
The dispatcher: unpacking, fan-out, concurrency, retries, and both entrypoints.

No docker, no cluster, no AWS. The three cloud backends are covered by
asserting on the call they *would* make — the argv, the RunTask kwargs, the Job
manifest — because that is where the bugs are: an environment variable in the
wrong flag, a task retrying twice because the orchestrator was also retrying.
The behaviour that has to be right regardless of backend is exercised against
`memory://`, and once for real against `process://`, which starts actual
child processes and reads actual exit codes.

Pure stdlib and no I/O beyond a temp directory.

    python tests/test_dispatch.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_embeddings.config import DispatchSettings, Settings      # noqa: E402
from rag_embeddings.queues import ParseRequest, open_queue, reset_memory  # noqa: E402
from rag_embeddings.runners import (                              # noqa: E402
    RecordingRunner,
    TaskSpec,
    TaskState,
    open_runner,
    task_name,
)
from rag_embeddings.runners.ecs import EcsRunner                  # noqa: E402
from rag_embeddings.runners.kube import KubernetesRunner          # noqa: E402
from rag_embeddings.runners.local import DockerRunner, ProcessRunner  # noqa: E402
from rag_embeddings.workers import lambda_dispatch                # noqa: E402
from rag_embeddings.workers.dispatcher import (                   # noqa: E402
    Dispatcher,
    documents_in,
    run as dispatch_run,
    spec_builder,
)


def settings_for(tmp: Path) -> Settings:
    return Settings.from_env(
        cache_dir=str(tmp / "cache"),
        parser_version="t",
        queue_url=f"file://{tmp / 'queue'}",
    )


def dispatcher_for(queue, runner, **kwargs):
    tmp = Path(kwargs.pop("tmp"))
    dispatch = DispatchSettings.from_env(
        runner_url="memory://", image="parser:test", **kwargs.pop("dispatch", {})
    )
    return Dispatcher(
        queue,
        runner,
        spec_builder(settings_for(tmp), dispatch),
        max_in_flight=kwargs.pop("max_in_flight", dispatch.max_in_flight),
        ack_on=kwargs.pop("ack_on", dispatch.ack_on),
        # Every status call counts in these tests; nothing may be throttled.
        status_interval=0.0,
        poll_interval=0.0,
        **kwargs,
    )


# ------------------------------------------------------------------ unpacking

def check_unpacking():
    """One message, N documents — four shapes, because four things publish."""
    single = documents_in({"uri": "inbox/a.pdf", "force": True})
    assert single == [ParseRequest(uri="inbox/a.pdf", force=True)], single

    batch = documents_in({"documents": [{"uri": "a.pdf"}, {"uri": "b.pdf"}]})
    assert [d.uri for d in batch] == ["a.pdf", "b.pdf"]

    shorthand = documents_in({"uris": ["a.pdf", "b.pdf"], "uri_prefix": "s3://bucket"})
    assert [d.uri for d in shorthand] == ["a.pdf", "b.pdf"]
    assert all(d.uri_prefix == "s3://bucket" for d in shorthand), (
        "the batch's shared options did not reach its documents"
    )

    # An S3 notification, which is what a bucket wired straight to the queue
    # sends. The key is url-encoded and '+' is a space, not a plus.
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "docs"},
                    "object": {"key": "reports/Q3+report+%282026%29.pdf"},
                },
            },
            {"eventName": "ObjectRemoved:Delete",
             "s3": {"bucket": {"name": "docs"}, "object": {"key": "old.pdf"}}},
        ]
    }
    from_s3 = documents_in(event)
    assert len(from_s3) == 1, "a deletion was turned into a parse"
    assert from_s3[0].uri == "s3://docs/reports/Q3 report (2026).pdf", from_s3[0].uri

    # The test notification S3 sends when the wiring is first configured.
    assert documents_in({"Records": [{"Event": "s3:TestEvent"}]}) == []

    for bad in ({}, {"documents": [{"mime": "application/pdf"}]}):
        try:
            documents_in(bad)
        except (KeyError, TypeError):
            pass
        else:
            raise AssertionError(f"a message with no uri must fail at the edge: {bad}")


# ---------------------------------------------------------------- task specs

def check_task_spec(tmp: Path):
    """What the container is told, and that it is told it the same way twice."""
    settings = settings_for(tmp)
    dispatch = DispatchSettings.from_env(
        runner_url="memory://",
        image="parser:1",
        task_command=["python", "-m", "parse"],
        task_env={"EXTRA": "1"},
        cpu=2, memory_mb=4096,
    )
    spec = spec_builder(settings, dispatch)(
        ParseRequest(uri="inbox/a.pdf", mime="application/pdf", force=True)
    )

    assert spec.image == "parser:1"
    assert list(spec.command) == ["python", "-m", "parse"]
    assert spec.cpu == 2 and spec.memory_mb == 4096
    assert spec.env["RAG_DOC_URI"] == "inbox/a.pdf"
    assert spec.env["RAG_DOC_MIME"] == "application/pdf"
    assert spec.env["RAG_DOC_FORCE"] == "1"
    assert spec.env["EXTRA"] == "1"
    # The pipeline's own configuration has to reach the container unchanged, or
    # a dispatched parse lands somewhere a compose-started one would not.
    assert spec.env["RAG_CACHE_DIR"] == str(settings.cache_dir)
    assert spec.env["RAG_PARSER_VERSION"] == "t"
    assert spec.env["RAG_QUEUE_URL"] == settings.queue_url
    # The whole request rides along, so the container has one thing to read.
    assert ParseRequest.from_body(json.loads(spec.env["RAG_PARSE_REQUEST"])).uri == (
        "inbox/a.pdf"
    )

    # A uri that is not a legal container name must still produce one, and two
    # tasks for the same document must not collide.
    name = task_name("s3://bucket/A Very Long Report Name (final) v2.PDF")
    assert len(name) <= 63 and name.islower()
    assert all(c.isalnum() or c == "-" for c in name), name
    assert task_name("a.pdf") != task_name("a.pdf")


# ------------------------------------------------------------------- routing

def check_open_runner(tmp: Path):
    """The url is the only place a backend is named."""
    from rag_embeddings.runners import reset_memory as reset_runners

    reset_runners()
    assert isinstance(open_runner("memory://"), RecordingRunner)
    assert open_runner("memory://") is open_runner("memory://"), (
        "a dry run could not inspect what it launched"
    )

    docker = open_runner("docker://?volume=/cache:/cache&volume=/q:/q&network=rag&keep=true")
    assert isinstance(docker, DockerRunner)
    assert docker.volumes == ["/cache:/cache", "/q:/q"] and docker.network == "rag"
    assert docker.keep is True

    process = open_runner(f"process://?log_dir={tmp}")
    assert isinstance(process, ProcessRunner)

    ecs = open_runner("ecs://prod/parse-task?subnets=subnet-a,subnet-b&region=us-east-1")
    assert isinstance(ecs, EcsRunner)
    assert ecs.cluster == "prod" and ecs.task_definition == "parse-task"
    assert ecs.subnets == ["subnet-a", "subnet-b"]

    k8s = open_runner("k8s://parsing?service_account=parser&ttl=60")
    assert isinstance(k8s, KubernetesRunner)
    assert k8s.namespace == "parsing" and k8s.service_account == "parser"
    assert k8s.ttl_seconds == 60

    try:
        open_runner("nomad://cluster")
    except NotImplementedError as exc:
        assert "nomad" in str(exc)
    else:
        raise AssertionError("unknown scheme must not silently fall back")

    # A typo in RAG_RUNNER_URL must be loud: containers with no volumes fail
    # one document at a time and look like a parser bug.
    try:
        open_runner("docker://?volumes=/cache:/cache")
    except ValueError as exc:
        assert "volumes" in str(exc)
    else:
        raise AssertionError("an unknown url option was silently ignored")

    # Fargate with no subnets cannot work; saying so here beats an AWS error
    # that does not mention subnets.
    try:
        open_runner("ecs://prod/task")
    except ValueError as exc:
        assert "subnet" in str(exc)
    else:
        raise AssertionError("FARGATE without subnets must be refused")


# ------------------------------------------------------- what each backend runs

def check_backend_calls():
    """The call each cloud backend would make, asserted without the cloud."""
    spec = TaskSpec(
        name="parse-a-pdf-1234",
        image="parser:1",
        command=("python", "-m", "parse"),
        env={"RAG_DOC_URI": "s3://b/a.pdf"},
        cpu=2, memory_mb=4096,
        labels={"rag.doc": "s3://b/a.pdf"},
    )

    argv = DockerRunner(volumes=["/cache:/cache"], network="rag").argv(spec)
    assert argv[:4] == ["docker", "run", "--detach", "--name"]
    assert "--rm" not in argv, (
        "--rm deletes the container before its exit code can be read"
    )
    assert argv[argv.index("--env") + 1] == "RAG_DOC_URI=s3://b/a.pdf"
    assert argv[argv.index("--volume") + 1] == "/cache:/cache"
    assert argv[argv.index("--cpus") + 1] == "2"
    assert argv[argv.index("--memory") + 1] == "4096m"
    # The image, then the command, and nothing after them.
    assert argv[-4:] == ["parser:1", "python", "-m", "parse"]

    fargate = EcsRunner("prod", "parse-task", subnets=["subnet-a"]).run_task_kwargs(spec)
    assert fargate["cluster"] == "prod" and fargate["taskDefinition"] == "parse-task"
    override = fargate["overrides"]["containerOverrides"][0]
    assert override["name"] == "parser"
    assert {"name": "RAG_DOC_URI", "value": "s3://b/a.pdf"} in override["environment"]
    assert override["command"] == ["python", "-m", "parse"]
    # Fargate sizes the task, not the container, and wants strings.
    assert fargate["overrides"]["cpu"] == "2048"
    assert fargate["overrides"]["memory"] == "4096"
    net = fargate["networkConfiguration"]["awsvpcConfiguration"]
    assert net["subnets"] == ["subnet-a"] and net["assignPublicIp"] == "DISABLED"

    ec2 = EcsRunner("prod", "parse-task", launch_type="EC2").run_task_kwargs(spec)
    assert "cpu" not in ec2["overrides"], "EC2 sizing belongs on the container"
    assert ec2["overrides"]["containerOverrides"][0]["memory"] == 4096

    job = KubernetesRunner("parsing", service_account="parser").job_body(spec)
    assert job["kind"] == "Job" and job["metadata"]["name"] == spec.name
    assert job["spec"]["backoffLimit"] == 0, (
        "kubernetes retrying as well as the queue means a poison document is "
        "attempted backoffLimit x max_attempts times"
    )
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["serviceAccountName"] == "parser"
    container = pod["containers"][0]
    assert container["image"] == "parser:1"
    assert container["args"] == ["python", "-m", "parse"], (
        "overriding command replaces the image's entrypoint"
    )
    assert container["resources"]["limits"] == {"cpu": "2000m", "memory": "4096Mi"}
    assert {"name": "RAG_DOC_URI", "value": "s3://b/a.pdf"} in container["env"]


# ---------------------------------------------------------------- the loop

def check_fan_out(tmp: Path):
    """Messages in, one container per document out, one ack per message."""
    reset_memory()
    queue = open_queue("memory://", "to-parse")
    for i in range(3):
        queue.publish(ParseRequest(uri=f"inbox/doc{i}.pdf").to_body())
    # And one message carrying a batch, which must not become one container.
    queue.publish({"uris": ["inbox/batch-a.pdf", "inbox/batch-b.pdf"]})

    runner = RecordingRunner("fanout")
    stats = dispatcher_for(queue, runner, tmp=tmp).run(idle_timeout=0.0)

    assert stats.received == 4 and stats.documents == 5, stats
    assert stats.launched == 5 and stats.succeeded == 5
    assert stats.acked == 4, "one ack per message, not per document"
    assert stats.ok and queue.depth() == 0
    assert sorted(runner.uris()) == [
        "inbox/batch-a.pdf", "inbox/batch-b.pdf",
        "inbox/doc0.pdf", "inbox/doc1.pdf", "inbox/doc2.pdf",
    ]
    assert len(runner.cleaned) == 5, "finished tasks were never cleaned up"


def check_concurrency_cap(tmp: Path):
    """A message holding a hundred documents must not start a hundred at once.

    This is the whole reason unpacked documents are queued rather than launched
    where they are found.
    """
    reset_memory()
    queue = open_queue("memory://", "flood")
    queue.publish({"uris": [f"doc{i}.pdf" for i in range(50)]})

    peak = {"n": 0}

    class Watching(RecordingRunner):
        def launch(self, spec):
            handle = super().launch(spec)
            peak["n"] = max(peak["n"], len(self._specs) - len(self.cleaned))
            return handle

    # Two polls before a task finishes, so tasks genuinely overlap.
    runner = Watching("flood", polls_before_done=2)
    stats = dispatcher_for(queue, runner, tmp=tmp, max_in_flight=4).run(idle_timeout=0.0)

    assert stats.launched == 50 and stats.acked == 1
    assert peak["n"] <= 4, f"ran {peak['n']} containers with max_in_flight=4"
    assert peak["n"] > 1, "nothing ran concurrently; the cap is not the limit"


def check_failure_is_retried_then_parked(tmp: Path):
    """A container that exits non-zero costs the message a retry, then a park."""
    reset_memory()
    queue = open_queue("memory://", "poison")
    queue.publish(ParseRequest(uri="inbox/bad.pdf").to_body())
    queue.publish(ParseRequest(uri="inbox/good.pdf").to_body())

    def outcome(spec):
        bad = spec.env["RAG_DOC_URI"].endswith("bad.pdf")
        return TaskState.FAILED if bad else TaskState.SUCCEEDED

    runner = RecordingRunner("poison", outcome=outcome)
    stats = dispatcher_for(queue, runner, tmp=tmp, max_attempts=2).run(idle_timeout=0.0)

    assert stats.succeeded == 1, "the good document behind it never ran"
    assert stats.dead_lettered == 1, "the failing document was retried forever"
    assert stats.retried == 1, "it was parked without a second attempt"
    assert not stats.ok
    assert queue.depth() == 0
    assert len(runner.launched) == 3, "expected 2 attempts at the bad document"
    assert [body for body, _reason in queue.dead][0]["uri"] == "inbox/bad.pdf"


def check_launch_refused_is_not_a_lost_document(tmp: Path):
    """An orchestrator that refuses to start a task must not cost a document."""
    reset_memory()
    queue = open_queue("memory://", "nocapacity")
    queue.publish(ParseRequest(uri="inbox/a.pdf").to_body())

    calls = {"n": 0}

    def on_launch(spec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("RunTask refused: no capacity")

    runner = RecordingRunner("nocapacity", on_launch=on_launch)
    stats = dispatcher_for(queue, runner, tmp=tmp, max_attempts=3).run(idle_timeout=0.0)

    assert stats.failed == 1 and stats.retried == 1
    assert stats.succeeded == 1, "the redelivered message was not dispatched"
    assert stats.acked == 1 and queue.depth() == 0


def check_ack_on_launch(tmp: Path):
    """`ack_on=launch` settles the message without waiting for the container."""
    reset_memory()
    queue = open_queue("memory://", "fire")
    queue.publish(ParseRequest(uri="inbox/a.pdf").to_body())

    # A task that never finishes: with ack_on=exit this loop would not return.
    runner = RecordingRunner("fire", polls_before_done=10**6)
    dispatcher = dispatcher_for(queue, runner, tmp=tmp, ack_on="launch")
    stats = dispatcher.run(idle_timeout=0.0)

    assert stats.launched == 1 and stats.acked == 1
    assert stats.succeeded == 0, "it should not have waited for an exit"
    assert queue.depth() == 0


def check_stop_signal_finishes_what_it_started(tmp: Path):
    """A stop must not abandon a claimed message or an orphan container."""
    reset_memory()
    queue = open_queue("memory://", "stopping")
    for i in range(3):
        queue.publish(ParseRequest(uri=f"inbox/doc{i}.pdf").to_body())

    checks = {"n": 0}

    def should_stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 1          # true from just after the first receive

    runner = RecordingRunner("stopping", polls_before_done=1)
    stats = dispatcher_for(queue, runner, tmp=tmp, max_in_flight=1).run(
        should_stop=should_stop
    )

    assert stats.stopped, "the loop did not report a signalled exit"
    assert stats.launched >= 1
    assert stats.acked == stats.received - stats.retried - stats.dead_lettered
    assert stats.succeeded == stats.launched, (
        "a container was left running when the dispatcher exited"
    )
    assert queue.depth() + stats.acked == 3, "the untouched backlog was disturbed"


def check_timeout_kills_the_container(tmp: Path):
    """A wedged container is killed and its document retried, not abandoned."""
    reset_memory()
    queue = open_queue("memory://", "wedged")
    queue.publish(ParseRequest(uri="inbox/slow.pdf").to_body())

    runner = RecordingRunner("wedged", polls_before_done=10**6)
    dispatcher = dispatcher_for(queue, runner, tmp=tmp, max_attempts=1, task_timeout=0.0)
    stats = dispatcher.run(idle_timeout=0.0)

    assert runner.cancelled, "the overrunning container was never killed"
    assert stats.failed == 1 and stats.dead_lettered == 1
    assert queue.depth() == 0


def check_real_processes(tmp: Path):
    """One end-to-end run against real child processes and real exit codes."""
    reset_memory()
    queue = open_queue("memory://", "processes")
    marker = tmp / "parsed"
    marker.mkdir()
    queue.publish({"uris": ["ok-1.pdf", "ok-2.pdf", "boom.pdf"]})

    # Stands in for the parser: writes a file named after the document it was
    # given, and exits 1 for the one that is meant to fail.
    program = (
        "import os,sys,pathlib;"
        "uri=os.environ['RAG_DOC_URI'];"
        f"pathlib.Path({str(marker)!r},uri).write_text(os.environ['RAG_PARSE_REQUEST']);"
        "sys.exit(1 if 'boom' in uri else 0)"
    )
    dispatch = DispatchSettings.from_env(
        runner_url=f"process://?log_dir={tmp / 'logs'}",
        task_command=[sys.executable, "-c", program],
        max_in_flight=2,
    )
    stats = dispatch_run(
        settings_for(tmp), dispatch, queue=queue, idle_timeout=0.0, max_attempts=1,
    )

    assert stats.launched == 3, stats
    assert stats.succeeded == 2 and stats.failed == 1
    assert stats.dead_lettered == 1, "the failing document was not parked"
    assert sorted(p.name for p in marker.iterdir()) == [
        "boom.pdf", "ok-1.pdf", "ok-2.pdf"
    ], "a container did not receive the document it was dispatched for"
    # The container was told which document it owns, in the form it will read.
    written = json.loads((marker / "ok-1.pdf").read_text())
    assert written["uri"] == "ok-1.pdf"


# ------------------------------------------------------------------- lambda

def check_lambda(tmp: Path, monkey_env):
    """An SQS event in, a partial batch response out.

    The response is the ack: anything not named in batchItemFailures is deleted
    from the queue, so a document that was never dispatched must appear there.
    """
    from rag_embeddings.runners import reset_memory as reset_runners

    reset_runners()
    runner = open_runner("memory://")
    runner.reset()
    runner.outcome = lambda spec: (
        TaskState.FAILED if "bad" in spec.env["RAG_DOC_URI"] else TaskState.SUCCEEDED
    )

    monkey_env({
        "RAG_RUNNER_URL": "memory://",
        "RAG_PARSER_IMAGE": "parser:test",
        "RAG_CACHE_DIR": str(tmp / "cache"),
        "RAG_ACK_ON": "exit",              # so a failing container is reported
        "RAG_MAX_IN_FLIGHT": "2",
    })

    event = {
        "Records": [
            {"messageId": "m1", "body": json.dumps({"uri": "inbox/a.pdf"})},
            {"messageId": "m2", "body": json.dumps({"uri": "inbox/bad.pdf"}),
             "attributes": {"ApproximateReceiveCount": "1"}},
            # One message, two documents: the S3 shape.
            {"messageId": "m3", "body": json.dumps({"Records": [
                {"eventName": "ObjectCreated:Put",
                 "s3": {"bucket": {"name": "docs"}, "object": {"key": "x.pdf"}}},
                {"eventName": "ObjectCreated:Put",
                 "s3": {"bucket": {"name": "docs"}, "object": {"key": "y.pdf"}}},
            ]})},
            {"messageId": "m4", "body": "not json at all"},
        ]
    }

    response = lambda_dispatch.handler(event, context=None)
    failed = {item["itemIdentifier"] for item in response["batchItemFailures"]}

    assert failed == {"m2", "m4"}, response
    assert sorted(runner.uris()) == [
        "inbox/a.pdf", "inbox/bad.pdf", "s3://docs/x.pdf", "s3://docs/y.pdf",
    ]

    # A direct invocation — the console's Test button — must work too.
    runner.reset()
    runner.outcome = lambda spec: TaskState.SUCCEEDED
    assert lambda_dispatch.handler({"uri": "inbox/direct.pdf"}) == {
        "batchItemFailures": []
    }
    assert runner.uris() == ["inbox/direct.pdf"]

    # And the deadline: a function about to time out must shorten its tasks
    # rather than be killed with the batch unreported.
    class Context:
        def get_remaining_time_in_millis(self):
            return 40_000

    trimmed = lambda_dispatch._fit_to_deadline(
        DispatchSettings.from_env(ack_on="exit", task_timeout=900), Context()
    )
    assert trimmed.task_timeout == 25.0, trimmed.task_timeout
    unchanged = lambda_dispatch._fit_to_deadline(
        DispatchSettings.from_env(ack_on="launch", task_timeout=900), Context()
    )
    assert unchanged.task_timeout == 900, "fire-and-forget does not wait"


def check_nothing_heavy_was_imported():
    """The dispatcher starts parsers; it must not be able to parse.

    It is the smallest process in the fan-out and the one that runs as a
    Lambda, where every dependency is a megabyte of zip and a slower cold
    start. This module has already imported the dispatcher, both entrypoints
    and all five backends, so if any of them reaches for a heavy import at
    module scope rather than inside the call that needs it, this fails.
    """
    for heavy in ("docling", "torch", "psycopg", "sentence_transformers",
                  "boto3", "kubernetes"):
        assert heavy not in sys.modules, (
            f"importing the dispatcher pulled in {heavy}"
        )


# --------------------------------------------------------------------- driver

def env_setter():
    """Set environment variables and undo them at the end of the run."""
    original = dict(__import__("os").environ)

    def apply(values):
        __import__("os").environ.update(values)

    def restore():
        environ = __import__("os").environ
        environ.clear()
        environ.update(original)

    return apply, restore


if __name__ == "__main__":
    apply_env, restore_env = env_setter()
    try:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            check_unpacking()
            check_task_spec(tmp)
            check_open_runner(tmp)
            check_backend_calls()
            check_fan_out(tmp)
            check_concurrency_cap(tmp)
            check_failure_is_retried_then_parked(tmp)
            check_launch_refused_is_not_a_lost_document(tmp)
            check_ack_on_launch(tmp)
            check_stop_signal_finishes_what_it_started(tmp)
            check_timeout_kills_the_container(tmp)
            check_real_processes(tmp)
            check_lambda(tmp, apply_env)
            check_nothing_heavy_was_imported()
    finally:
        restore_env()
    print("dispatch ok")
