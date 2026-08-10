"""
The queue contract, run against every backend.

One set of assertions, parametrised over the backends, because that is the
claim the Strategy is making: swapping the backend does not change the
semantics. A backend that passes this is one the workers can run on unchanged.

Pure stdlib and no I/O beyond a temp directory, so it runs in well under a
second and belongs on every edit.

    python tests/test_queues.py
"""

import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_embeddings.queues import open_queue, reset_memory      # noqa: E402
from rag_embeddings.queues.files import FileQueue               # noqa: E402
from rag_embeddings.queues.memory import InMemoryQueue          # noqa: E402


def backends(tmp: Path):
    """Every backend, each as a fresh factory taking a queue name."""
    reset_memory()
    yield "memory", lambda name, **kw: InMemoryQueue(name, **kw)
    yield "file", lambda name, **kw: FileQueue(tmp / name, name, **kw)


# ------------------------------------------------------------------ contract

def check_publish_receive_ack(new):
    q = new("basic")
    assert q.depth() == 0
    assert q.receive() is None, "empty queue must not hand out a message"

    q.publish({"uri": "a.pdf"})
    q.publish({"uri": "b.pdf"})
    assert q.depth() == 2

    first = q.receive()
    assert first is not None and first.body["uri"] == "a.pdf", "not FIFO"
    assert first.attempts == 1 and not first.redelivered

    # Claimed messages are invisible to everyone else.
    assert q.depth() == 1
    second = q.receive()
    assert second.body["uri"] == "b.pdf"
    assert q.receive() is None, "handed out a message twice"

    q.ack(first)
    q.ack(second)
    assert q.depth() == 0
    assert q.receive() is None, "acked message came back"


def check_nack_redelivers(new):
    q = new("retry")
    q.publish({"n": 1})

    first = q.receive()
    q.nack(first)
    assert q.depth() == 1, "nacked message did not become visible again"

    second = q.receive()
    assert second.body == {"n": 1}
    assert second.attempts == 2, f"attempts not counted: {second.attempts}"
    assert second.redelivered


def check_visibility_timeout(new):
    """A worker that dies holding a claim must not strand the message."""
    q = new("crash", visibility_timeout=0.0)
    q.publish({"n": 1})

    claimed = q.receive()
    assert claimed is not None
    # Never acked, never nacked — the worker is gone.
    reclaimed = q.receive()
    assert reclaimed is not None, "timed-out claim was never reclaimed"
    assert reclaimed.attempts == 2


def check_consume_acks_and_dead_letters(new):
    q = new("consume")
    for i in range(3):
        q.publish({"n": i})

    seen = []
    stats = q.consume(lambda body: seen.append(body["n"]), idle_timeout=0.0)
    assert seen == [0, 1, 2], seen
    assert stats.received == 3 and stats.acked == 3 and stats.ok
    assert q.depth() == 0

    # A handler that always raises retries to the limit, then parks it.
    poison = new("poison")
    poison.publish({"bad": True})
    tries = []

    def boom(body):
        tries.append(body)
        raise ValueError("nope")

    stats = poison.consume(boom, idle_timeout=0.0, max_attempts=3)
    assert len(tries) == 3, f"expected 3 attempts, got {len(tries)}"
    assert stats.dead_lettered == 1 and stats.failed == 3
    assert not stats.ok
    assert poison.depth() == 0, "dead-lettered message is still on the queue"
    assert "nope" in stats.errors[0]


def check_failure_does_not_stop_the_worker(new):
    """One poison document must not cost the rest of the backlog."""
    q = new("mixed")
    for i in range(4):
        q.publish({"n": i})

    done = []

    def handler(body):
        if body["n"] == 1:
            raise RuntimeError("bad document")
        done.append(body["n"])

    stats = q.consume(handler, idle_timeout=0.0, max_attempts=2)
    assert sorted(done) == [0, 2, 3], done
    assert stats.dead_lettered == 1
    assert q.depth() == 0


def check_max_messages(new):
    q = new("bounded")
    for i in range(5):
        q.publish({"n": i})

    stats = q.consume(lambda body: None, max_messages=2, idle_timeout=0.0)
    assert stats.acked == 2
    assert q.depth() == 3, "max_messages consumed more than it was allowed"


def check_exclusive_delivery(new):
    """Concurrent consumers, no coordination: every message goes to exactly one.

    This is the property the whole fan-out rests on. For FileQueue it is really
    a test of the rename claim, which is why it runs with threads rather than
    being asserted in a comment.
    """
    q = new("race")
    total = 60
    for i in range(total):
        q.publish({"n": i})

    got: list[list[int]] = [[] for _ in range(6)]

    def worker(slot):
        q.consume(lambda body: got[slot].append(body["n"]), idle_timeout=0.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    delivered = [n for slot in got for n in slot]
    assert len(delivered) == total, f"lost or duplicated: {len(delivered)}/{total}"
    assert sorted(delivered) == list(range(total)), "a message went to two workers"
    assert q.depth() == 0


CHECKS = [
    check_publish_receive_ack,
    check_nack_redelivers,
    check_visibility_timeout,
    check_consume_acks_and_dead_letters,
    check_failure_does_not_stop_the_worker,
    check_max_messages,
    check_exclusive_delivery,
]


# ------------------------------------------------------------------- routing

def check_open_queue(tmp: Path):
    """The uri is the only place a backend is named."""
    reset_memory()
    assert isinstance(open_queue("memory://", "x"), InMemoryQueue)
    # memory:// has to be a registry or a producer and consumer in one process
    # would each get their own queue.
    assert open_queue("memory://", "x") is open_queue("memory://", "x")
    assert open_queue("memory://", "y") is not open_queue("memory://", "x")

    assert isinstance(open_queue(f"file://{tmp}", "x"), FileQueue)
    assert isinstance(open_queue(str(tmp), "x"), FileQueue)
    assert isinstance(open_queue(tmp, "x"), FileQueue)

    # A file:// queue is shared through the filesystem, not a registry.
    a = open_queue(f"file://{tmp}", "shared")
    b = open_queue(f"file://{tmp}", "shared")
    a.publish({"n": 1})
    assert b.receive().body == {"n": 1}, "two handles disagreed about one queue"

    try:
        open_queue("sqs://prod", "x")
    except NotImplementedError as exc:
        assert "sqs" in str(exc)
    else:
        raise AssertionError("unknown scheme must not silently fall back")


def check_message_types():
    """Round-trip through JSON-shaped dicts, which is what a broker stores."""
    import json

    from rag_embeddings.queues import IndexRequest, ParseRequest

    parse = ParseRequest(uri="inbox/a.pdf", uri_prefix="s3://bucket")
    assert ParseRequest.from_body(json.loads(json.dumps(parse.to_body()))) == parse
    assert ParseRequest.from_body({"uri": "x"}).force is False

    index = IndexRequest(
        sha256="abc", uri="s3://bucket/a.pdf", mime="application/pdf",
        parser_version="v1", parsed_at="2026-08-08T00:00:00+00:00",
        with_tables=False,
    )
    assert IndexRequest.from_body(json.loads(json.dumps(index.to_body()))) == index
    assert index.with_tables is False and index.with_chunks is True

    try:
        IndexRequest.from_body({"sha256": "abc"})
    except KeyError:
        pass
    else:
        raise AssertionError("a truncated message must fail at the edge")


def check_queues_import_without_a_parser():
    """`import rag_embeddings.queues` must not drag docling in.

    The producer is a shell loop or an S3 event handler. If publishing a
    filename required the parser, every one of those would need the 2 GB image.
    """
    assert "docling" not in sys.modules, (
        "importing the queue package pulled in the parser"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for label, new in backends(tmp):
            for check in CHECKS:
                check(new)
            print(f"  {label:8} {len(CHECKS)} checks ok")
        check_open_queue(tmp / "routing")
        check_message_types()
        check_queues_import_without_a_parser()
    print("queues ok")
