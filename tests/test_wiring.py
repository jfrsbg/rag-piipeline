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
from rag_embeddings.profiles import resolve                   # noqa: E402
from rag_embeddings.steps.index import main as index_main     # noqa: E402
from rag_embeddings.steps.parse import main as parse_main     # noqa: E402


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


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check(Path(tmp))
    print("wiring ok")
