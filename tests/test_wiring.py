"""
Wiring check for the two steps, with the heavy dependencies stubbed out.

Covers step 1 -> cache -> step 2 -> SQL: import paths, CLI flags, the manifest
round-trip between the two processes, and the statement order inside the
transaction. Runs in under a second and needs neither torch nor a database,
which is the point — it is the test you can run on every edit.

    python tests/test_wiring.py
"""

import json
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


# --------------------------------------------------------------------- docling

class Prov:
    page_no = 3


class Cell:
    def __init__(self, text, row, col, col_header=False):
        self.text = text
        self.start_row_offset_idx, self.end_row_offset_idx = row, row + 1
        self.start_col_offset_idx, self.end_col_offset_idx = col, col + 1
        self.column_header, self.row_header = col_header, False


class TableData:
    num_rows, num_cols = 2, 2
    table_cells = [
        Cell("h1", 0, 0, True), Cell("h2", 0, 1, True),
        Cell("v1", 1, 0), Cell("v2", 1, 1),
    ]


class Table:
    self_ref = "#/tables/0"
    data = TableData()
    prov = [Prov()]

    def caption_text(self, doc):
        return "a caption"

    def export_to_markdown(self, doc):
        return "| h1 | h2 |"


class FakeDoc:
    tables = [Table()]

    def save_as_json(self, path):
        Path(path).write_text(json.dumps({"fake": True}))

    @classmethod
    def load_from_json(cls, path):
        assert json.loads(Path(path).read_text())["fake"]
        return cls()


class FakeConverter:
    calls = []

    def convert(self, source):
        FakeConverter.calls.append(source)
        return types.SimpleNamespace(document=FakeDoc())


# --------------------------------------------------------- chunker / embedder

class Item:
    self_ref = "#/texts/0"
    prov = [Prov()]


class Meta:
    headings = ["H1", "H2"]
    doc_items = [Item()]


class Chunk:
    text = "chunk body"
    meta = Meta()


class FakeChunker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chunk(self, dl_doc):
        return iter([Chunk(), Chunk()])

    def contextualize(self, chunk):
        return "H1\nH2\n" + chunk.text


class FakeTokenizer:
    seen = []

    @classmethod
    def from_pretrained(cls, model_name, max_tokens=None):
        cls.seen.append((model_name, max_tokens))
        return cls()

    def count_tokens(self, text):
        return len(text.split())


class FakeModel:
    loaded = []

    def __init__(self, model_id):
        FakeModel.loaded.append(model_id)

    def encode(self, texts, **kwargs):
        # the real encode() returns ndarray rows, which carry .tolist()
        return [types.SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3]) for _ in texts]


# --------------------------------------------------------------------- psycopg

class FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, query, params=None):
        self.log.append((" ".join(query.split()), params))
        return self

    def executemany(self, query, rows):
        self.log.append((" ".join(query.split()), f"<{len(rows)} rows>"))

    def fetchone(self):
        last = self.log[-1][0]
        if last.startswith("insert into documents") or last.startswith("select id"):
            return (42,)
        return None

    def fetchall(self):
        return []


class FakeConn:
    def __init__(self):
        self.log = []

    def cursor(self):
        return FakeCursor(self.log)

    def execute(self, query, params=None):
        return FakeCursor(self.log).execute(query, params)

    @contextmanager
    def transaction(self):
        self.log.append(("BEGIN", None))
        yield
        self.log.append(("COMMIT", None))

    def close(self):
        pass


CONN = FakeConn()

stub("docling")
stub("docling.document_converter", DocumentConverter=FakeConverter)
stub("docling.chunking", HybridChunker=FakeChunker)
stub("docling_core")
stub("docling_core.types")
stub("docling_core.types.doc", DoclingDocument=FakeDoc)
stub("docling_core.transforms")
stub("docling_core.transforms.chunker")
stub("docling_core.transforms.chunker.tokenizer")
stub(
    "docling_core.transforms.chunker.tokenizer.huggingface",
    HuggingFaceTokenizer=FakeTokenizer,
)
stub("sentence_transformers", SentenceTransformer=FakeModel)
stub("psycopg", connect=lambda dsn, autocommit=False: CONN)
stub("psycopg.types")
stub("psycopg.types.json", Jsonb=lambda value: ("jsonb", value))
stub("pgvector")
stub("pgvector.psycopg", register_vector=lambda conn: None)

sys.path.insert(0, str(ROOT))

import rag_embeddings                                        # noqa: E402
from rag_embeddings.cache import Manifest, cached_shas         # noqa: E402
from rag_embeddings.config import Settings                     # noqa: E402
from rag_embeddings.profiles import resolve                    # noqa: E402
from rag_embeddings.queues import IndexRequest, open_queue     # noqa: E402
from rag_embeddings.steps.index import main as index_main      # noqa: E402
from rag_embeddings.steps.parse import main as parse_main      # noqa: E402
from rag_embeddings.workers import enqueue as producer         # noqa: E402
from rag_embeddings.workers import index_worker, parse_worker  # noqa: E402


def check(work: Path) -> None:
    inbox, cache = work / "inbox", work / "cache"
    inbox.mkdir()
    (inbox / "report.pdf").write_bytes(b"%PDF-1.4 fake bytes")
    (inbox / "notes.md").write_text("# hello")

    common = ["--cache-dir", str(cache), "--log-level", "WARNING"]

    # -- step 1 ------------------------------------------------------------
    assert parse_main([str(inbox), *common, "--parser-version", "t"]) == 0
    names = sorted(p.name for p in cache.iterdir())
    assert len(names) == 4, names                       # parse + manifest each
    assert sum(n.endswith(".meta.json") for n in names) == 2

    manifest = json.loads(next(cache.glob("*.meta.json")).read_text())
    assert manifest["parser_version"] == "t"
    assert manifest["mime"] in {"application/pdf", "text/markdown"}

    # a second run over the same bytes must not re-parse
    parsed = len(FakeConverter.calls)
    parse_main([str(inbox / "report.pdf"), *common])
    assert len(FakeConverter.calls) == parsed, "cache hit still re-parsed"

    # -- step 2, both branches --------------------------------------------
    CONN.log.clear()
    assert index_main([*common, "--dsn", "x", "--parser-version", "t"]) == 0

    order = [q for q, _ in CONN.log]
    assert order.count("BEGIN") == order.count("COMMIT") == 2
    first = order[: order.index("COMMIT") + 1]
    assert first[0] == "BEGIN"
    assert first[1].startswith("insert into documents")
    assert any(q.startswith("insert into doc_tables") for q in first)
    assert any(q.startswith("insert into chunks") for q in first)

    # -- step 2, one branch at a time -------------------------------------
    CONN.log.clear()
    loaded = len(FakeModel.loaded)
    index_main([*common, "--dsn", "x", "--tables-only"])
    assert len(FakeModel.loaded) == loaded, "--tables-only loaded a model"
    assert not any("chunks" in q for q, _ in CONN.log)
    assert any("delete from doc_tables" in q for q, _ in CONN.log)

    CONN.log.clear()
    index_main([*common, "--dsn", "x", "--chunks-only"])
    assert not any("doc_tables" in q for q, _ in CONN.log)
    assert any("insert into chunks" in q for q, _ in CONN.log)

    # -- library surface ---------------------------------------------------
    for name in ("ingest", "reextract", "rechunk", "stale_documents",
                 "extract_tables", "build_chunks", "write_all", "connect",
                 "Embedder", "EmbedProfile"):
        assert hasattr(rag_embeddings, name), f"{name} no longer exported"

    # -- profiles ----------------------------------------------------------
    assert resolve("e5-large").passage_prefix == "passage: "
    assert resolve("bge-m3", max_tokens=2048).version == "BAAI/bge-m3@2048-128"
    assert resolve("some/model", max_tokens=512).model_id == "some/model"
    try:
        resolve("some/model")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown profile without max_tokens must raise")

    # headroom is what the chunker is sized with, not the profile limit
    assert FakeTokenizer.seen[0] == ("BAAI/bge-m3", 8192 - 128)


def check_workers(work: Path) -> None:
    """The same two steps over a queue: producer -> parse -> index -> SQL.

    Runs both workers in this process against a file-backed queue, which is
    the arrangement compose uses — only the container boundary is missing.
    """
    inbox, cache, queue_root = work / "inbox", work / "cache", work / "queue"
    inbox.mkdir()
    for i in range(3):
        (inbox / f"doc{i}.pdf").write_bytes(f"%PDF-1.4 doc {i}".encode())

    settings = Settings.from_env(
        cache_dir=str(cache),
        parser_version="w",
        dsn="x",
        queue_url=f"file://{queue_root}",
    )
    to_parse = open_queue(settings.queue_url, settings.parse_queue)
    to_index = open_queue(settings.queue_url, settings.index_queue)

    # -- producer ----------------------------------------------------------
    assert len(producer.enqueue_files([str(inbox)], settings)) == 3
    assert to_parse.depth() == 3, "producer published one message per document"

    # -- step 1 workers ----------------------------------------------------
    # Two of them, sharing the queue with no coordination, exactly as two pods
    # sharing a volume would. idle_timeout=0 is what makes them exit.
    first = parse_worker.run(settings, idle_timeout=0.0)
    second = parse_worker.run(settings, idle_timeout=0.0)
    assert first.acked + second.acked == 3, "documents were lost or duplicated"
    assert second.received == 0, "the drained queue handed out a message twice"
    assert to_parse.depth() == 0
    assert to_index.depth() == 3, "parses were not announced downstream"

    # The manifest rides on the message; step 2 never has to read the sidecar.
    peeked = IndexRequest.from_body(to_index.receive().body)
    assert peeked.parser_version == "w" and peeked.mime == "application/pdf"
    assert peeked.to_manifest() == Manifest.read(peeked.sha256, cache)

    # -- step 2 worker -----------------------------------------------------
    CONN.log.clear()
    loaded = len(FakeModel.loaded)
    stats = index_worker.run(settings, conn=CONN, idle_timeout=0.0)

    assert stats.acked == 2, stats            # one was consumed by the peek above
    assert len(FakeModel.loaded) == loaded + 1, (
        "the model was loaded per message; it must be loaded per worker"
    )
    order = [q for q, _ in CONN.log]
    assert order.count("BEGIN") == order.count("COMMIT") == 2, "one commit per doc"
    assert any(q.startswith("insert into doc_tables") for q in order)
    assert any(q.startswith("insert into chunks") for q in order)

    # -- a poison message must not take the worker down --------------------
    to_index.publish(
        IndexRequest(
            sha256="0" * 64, uri="gone.pdf", mime="application/pdf",
            parser_version="w", parsed_at="2026-08-08T00:00:00+00:00",
        ).to_body()
    )
    to_index.publish(peeked.to_body())                     # a good one behind it

    stats = index_worker.run(settings, conn=CONN, idle_timeout=0.0, max_attempts=2)
    assert stats.dead_lettered == 1, "the missing parse was retried forever"
    assert stats.acked == 1, "the good document behind it never ran"
    assert to_index.depth() == 0

    # -- re-enqueueing from the cache is a coordinator job, not a worker's --
    assert sorted(producer.enqueue_cached(None, settings, queue=to_index)) == sorted(
        cached_shas(cache)
    )
    assert to_index.depth() == 3


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        check_workers(Path(tmp))
    print("wiring ok")
