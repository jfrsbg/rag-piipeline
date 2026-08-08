# rag-embeddings

Parses documents once, caches the parse, and fans the result out to two sinks:
tables into relational columns, everything else into chunk embeddings. Both
writes land in a single transaction.

```mermaid
flowchart LR
    subgraph step1 ["step 1 — parse and cache"]
        direction LR
        src[(object storage)] --> parse[parse]
        parse --> cache["cache/parsed/<br>{sha}.json + {sha}.meta.json"]
    end

    subgraph step2 ["step 2 — extract and store"]
        direction LR
        tables["doc.tables"] --> doc_tables[(doc_tables)]
        chunk["chunker.chunk()"] --> chunks[(chunks + embeddings)]
        doc_tables -.-> commit(["one commit"])
        chunks -.-> commit
    end

    cache --> tables
    cache --> chunk
```

The two steps run as separate processes. Step 1 needs Docling and a disk; step 2
needs a GPU-ish machine and the database. Neither needs what the other has.

---

## Layout

```
step1_parse.py               entrypoint: parse and cache
step2_index.py               entrypoint: extract and store
rag_embeddings/
  config.py                  env -> Settings, the only place os.environ is read
  profiles.py                EmbedProfile + the named profiles
  embedder.py                tokenizer, chunker and model, from one profile
  cache.py                   parse cache and the manifest sidecar
  cli.py                     flags shared by both steps
  extraction/
    tables.py                branch A: tables -> relational rows
    chunks.py                branch B: chunks -> vectors
  storage/
    connection.py            connect() + register_vector
    sql.py                   every statement, in one place
    writer.py                write_all(): both branches, one commit
  steps/
    parse.py                 step 1
    index.py                 step 2
  pipeline.py                ingest / reextract / rechunk, for in-process use
sql/schema.sql               applied by the db container on first start
tests/test_wiring.py         both steps end to end, no torch, no database
```

Each entrypoint file does nothing but call `main()` in its step module, so the
container invocation and the importable function never drift apart.

---

## Run it with Docker

```bash
docker compose up -d db                      # postgres + pgvector, schema applied
docker compose run --rm parse /data/inbox    # step 1
docker compose run --rm index                # step 2
```

`./inbox` is mounted read-only at `/data/inbox` and `./cache` holds the parse
cache, so both steps see the same files you do. Model weights land in a named
volume — a rebuild does not re-download BGE-M3.

Arguments after the service name go to the step:

```bash
docker compose run --rm parse /data/inbox --pattern '*.pdf' --force
docker compose run --rm index --tables-only 9f2a...c1
docker compose run --rm index --stale
```

Without compose:

```bash
docker build -t rag-embeddings .
docker run --rm -v ./cache:/cache -v ./inbox:/data/inbox:ro \
  rag-embeddings step1_parse.py /data/inbox
docker run --rm -v ./cache:/cache -v hf-models:/models \
  -e RAG_DSN=postgresql://user:pass@host/docs \
  rag-embeddings step2_index.py
```

The image installs the CPU torch wheel by default. For a GPU host:
`docker build --build-arg TORCH_EXTRA=cu126 .`

Rebuilds are cheap in two ways worth knowing about. The dependency layer is
keyed on `pyproject.toml` + `uv.lock` only, so editing a runtime setting or any
source file does not touch it; and when the lock *does* change, the wheels come
from a BuildKit cache mount instead of the network. That cache lives outside the
image — `docker builder prune` is what clears it, and a cold CI runner has no
such cache and will download the full ~2 GB.

Model weights are a separate cache: they live on the `hf-models` volume, not in
the image, so they survive rebuilds but not `docker compose down -v`. To bake
them in instead — for CI, an air-gapped host, or an image you ship to someone
else — build with `--build-arg PREFETCH_MODELS=1`. That adds ~2.6 GB to the
image, and is redundant locally where the volume already holds them.

---

## Run it locally

```bash
uv sync --extra cpu                     # or --extra cu126 on a GPU host
uv run step1_parse.py inbox/
uv run step2_index.py --dsn postgresql://localhost/docs
```

First run downloads the Docling layout models and the embedding model (~2 GB for
BGE-M3). CPU works; a GPU makes the initial backfill hours instead of days.

`uv sync` also puts `rag-parse` and `rag-index` in `.venv/bin` — the same two
`main()` functions the step files call.

`uv.lock` is committed and is the single source of truth for versions; the
`cpu` and `cu126` extras are declared conflicting, so exactly one may be
selected. There is no `requirements.txt` — `pyproject.toml` replaced it.

---

## Configuration

Every knob is a CLI flag with an environment-variable default, resolved once in
`config.py`. Flags beat the environment; the environment beats the defaults.

| Env var | Flag | Default |
|---|---|---|
| `RAG_CACHE_DIR` | `--cache-dir` | `cache/parsed` |
| `RAG_PARSER_VERSION` | `--parser-version` | `docling-2.118` |
| `RAG_DSN` (or `DATABASE_URL`) | `--dsn` | `postgresql://localhost/docs` |
| `RAG_EMBED_PROFILE` | `--profile` | `bge-m3` |
| `RAG_EMBED_MAX_TOKENS` | `--max-tokens` | from the profile |
| `RAG_EMBED_HEADROOM` | `--headroom` | `128` |

Everything model-related still lives in one frozen dataclass. The tokenizer, the
chunker, and the embedder are all derived from it, so a mismatch between chunk
sizing and embedding capacity requires constructing two profiles — which shows
up in a diff.

```python
from rag_embeddings import EmbedProfile, Embedder

BGE_M3 = EmbedProfile(model_id="BAAI/bge-m3", max_tokens=8192)
```

`--profile` takes a name from `profiles.py` (`bge-m3`, `e5-large`) or any
HuggingFace model id, in which case `--max-tokens` is required — there is no
safe default to guess. E5-family models need their prefixes; omitting them
degrades retrieval quietly rather than loudly, which is why `e5-large` is a named
profile rather than a flag you have to remember.

`max_tokens` comes from the model card, deliberately not from
`tokenizer.model_max_length` — many HF configs ship a sentinel in the range of
10^19 there, and feeding that to the chunker means it never splits anything.

`headroom` (default 128) sizes the chunker below the real limit because
`contextualize()` prepends the heading path *after* the split decision was made.

---

## Step 1 — parse and cache

```bash
python step1_parse.py inbox/                      # a directory
python step1_parse.py inbox/ --pattern '*.pdf'    # filtered
python step1_parse.py a.pdf b.pdf 'reports/*.docx'
python step1_parse.py inbox/ --force              # after a Docling upgrade
```

Writes two files per document: `{sha}.json` (the parse) and `{sha}.meta.json`
(uri, mime, parser version, timestamp). Prints `sha<TAB>uri` per document, so it
pipes.

The manifest exists because the steps are separate processes. Step 2 only gets a
sha; rather than have it re-derive the uri from a source that may no longer be
reachable, step 1 records what it knew at parse time.

Content-addressed and idempotent: re-running over the same bytes is a cache hit,
so a crashed batch restarts from the top without duplicates or cleanup.

For object storage, stage the bytes to disk and keep the real location:

```bash
aws s3 sync s3://bucket/prefix ./inbox
python step1_parse.py inbox/ --uri-prefix s3://bucket/prefix
```

That is what ends up in `documents.uri`, while Docling opens the local file.

---

## Step 2 — extract and store

```bash
python step2_index.py                    # everything in the cache
python step2_index.py 9f2a...c1 4b81...0e
python step2_index.py --skip-existing    # only what isn't in the database yet
```

Loads each cached parse, runs both branches, and writes them in one transaction
per document. No parser runs here.

The branch flags are the reprocessing triggers, and they are why the two
branches are separate code paths at all:

| Changed | Run | Re-parses? | Loads a model? |
|---|---|---|---|
| Extraction logic / table schema | `step2_index.py --tables-only` | no | no |
| Embedding model, `max_tokens`, chunker | `step2_index.py --chunks-only` | no | yes |
| Docling version, OCR settings | `step1_parse.py --force` then step 2 | yes | yes |

After changing the profile, find exactly what is stale rather than reprocessing
everything:

```bash
python step2_index.py --stale --chunks-only --profile bge-m3 --max-tokens 2048
```

`--stale` compares each chunk's stored `chunk_config` against the active profile
version, so it catches model swaps and token-limit changes alike.

---

## Using it as a library

`pipeline.py` keeps the original in-process entry points for callers that want
parse and store in one go — `ingest` is step 1 followed by step 2:

```python
from pathlib import Path
from rag_embeddings import Embedder, Settings, connect, ingest, rechunk, stale_documents

settings = Settings.from_env()
emb = Embedder(settings.profile)
conn = connect(settings.dsn)

for path in Path("inbox").glob("*.pdf"):
    ingest(conn, str(path), path.read_bytes(), mime="application/pdf", emb=emb)

for sha in stale_documents(conn, emb):
    rechunk(conn, sha, emb)
```

---

## Database setup

`sql/schema.sql` holds the DDL; the `db` service applies it on first start of an
empty data directory. Applying it by hand:

```bash
psql postgresql://localhost/docs -f sql/schema.sql
```

`vector(1024)` matches BGE-M3. Change it if you change models — and note that
altering the dimension requires dropping and rebuilding the column.

Before building the HNSW index on a large table, raise `maintenance_work_mem`
(1–2 GB) or the build will take far longer than it should.

---

## Querying

**Semantic search.** Embed the query through the same profile — same model, same
normalization, same prefix:

```python
qvec = emb.encode_query("what were the Q3 logistics costs?")

rows = conn.execute(
    """select id, document_id, ord, text, page
       from chunks order by embedding <=> %s::vector limit 20""",
    (qvec,),
).fetchall()
```

**Widen the context after retrieval.** Chunks are stored without overlap; the
neighbours are one indexed lookup away, which is why `ord` exists:

```python
window = conn.execute(
    """select text from chunks
       where document_id = %s and ord between %s - 1 and %s + 1
       order by ord""",
    (document_id, ord_, ord_),
).fetchall()
```

**Follow a table chunk to its authoritative cells.** A hit whose `refs` contains
a table's `self_ref` resolves to the exact grid:

```sql
select t.cells, t.columns, t.page
from chunks c
join doc_tables t
  on t.document_id = c.document_id
 and c.refs @> to_jsonb(t.self_ref)
where c.id = %s;
```

That's the point of sending tables to both sinks — search finds the readable
markdown copy, the application reads the typed values.

---

## Things that will bite you

**A missing manifest fails step 2, not step 1.** `{sha}.meta.json` is written
after the parse. If you copy a cache directory around, copy both files, or step 2
raises `no manifest for {sha}` — deliberately, rather than inventing a uri.

**Overflow warnings are real.** `build_chunks` logs `chunk overflow` when the
contextualized text exceeds the profile limit. It does not truncate or split — it
records `token_count` so you can find them later:

```sql
select document_id, ord, token_count from chunks where token_count > 8192;
```

If that returns rows, either raise `headroom` or add a hard splitter for those
cases. Wide tables and deep heading paths are the usual causes.

**`ord` must stay dense.** It's assigned by `enumerate` over the chunker's
output, and the emission order *is* reading order. If you add filtering (dropping
empty or boilerplate chunks), filter after assignment or the windowed reads above
will silently return short.

**Table cells, not DataFrames.** `extract_tables` reads `table.data.table_cells`
because `export_to_dataframe()` has a known class of bug where a column vanishes
from the export while the data sits intact in the JSON. If you need a frame for
analysis, build it from `cells` yourself.

**`prov` can be empty.** Merged list groups and some HTML-derived nodes carry no
provenance, so `page` will be `null` for them. That's expected, not a failure —
don't add a NOT NULL constraint there.

**Both branches delete before insert.** `write_all`, and therefore every step 2
run, clears the document's existing rows first, inside the transaction.
Reprocessing is a replace, not an append.

**Don't pin `transformers` yourself.** `docling-core[chunking]` caps it at `<5.9`
on macOS and `<6` elsewhere; adding a third opinion is how you get an unsolvable
resolve on one platform and a silently different version on the other. See the
comment on `dependencies` in `pyproject.toml`.

---

## Tests

```bash
python tests/test_wiring.py
```

Stubs Docling, sentence-transformers and psycopg, then runs step 1 into a temp
cache and step 2 out of it, asserting the manifest round-trip, the statement
order inside the transaction, and that `--tables-only` never loads a model. No
torch, no database, under a second.

---

## Extending

The obvious next stage is the extraction pass — a schema-driven read over the
chunks (LangExtract, Instructor) producing typed rows in a `facts` table, plus
the entity resolution described in `entity-resolution-design.md`. It belongs as a
third step reading the same cache, for the same reason step 2 does: a change to
your extraction schema shouldn't cost a re-parse.
