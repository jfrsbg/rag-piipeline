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

That split is also the fan-out boundary: the same two steps run as pools of
queue workers, one document per message, without changing what the diagram
above describes. See [Running it in parallel](#running-it-in-parallel).

---

## Layout

```
step1_parse.py               entrypoint: parse and cache
step2_index.py               entrypoint: extract and store
enqueue.py                   entrypoint: publish work onto a queue
worker_parse.py              entrypoint: step 1 as a queue consumer
worker_index.py              entrypoint: step 2 as a queue consumer
rag_embeddings/
  config.py                  env -> Settings, the only place os.environ is read
  profiles.py                EmbedProfile + the named profiles
  embedder.py                tokenizer, chunker and model, from one profile
  blobstore.py               where the cache lives: local now, object store later
  cache.py                   parse cache and the manifest sidecar
  cli.py                     flags shared by the steps and the workers
  extraction/
    tables.py                branch A: tables -> relational rows
    chunks.py                branch B: chunks -> vectors
  storage/
    connection.py            connect() + register_vector
    sql.py                   every statement, in one place
    writer.py                write_all(): both branches, one commit
  steps/
    parse.py                 step 1, batch over a directory
    index.py                 step 2, batch over the cache
  queues/
    base.py                  the Queue interface + consume/retry/dead-letter
    memory.py                backend: in-process, for tests
    files.py                 backend: a directory, for containers on a volume
    messages.py              ParseRequest and IndexRequest
  workers/
    enqueue.py               the producer, and the whole-cache coordinator jobs
    parse_worker.py          step 1, one document per message
    index_worker.py          step 2, one document per message
  pipeline.py                ingest / reextract / rechunk, for in-process use
sql/schema.sql               applied by the db container on first start
tests/test_wiring.py         both steps and both workers, no torch, no database
tests/test_queues.py         the queue contract, run against every backend
```

Each entrypoint file does nothing but call `main()` in its step or worker
module, so the container invocation and the importable function never drift
apart.

`rag_embeddings/__init__.py` resolves its exports lazily. Importing
`rag_embeddings.queues` is all a producer needs to put a filename on a queue,
and eager exports would have made it pay for docling and torch on the way in —
on the smallest and most-replicated container in the fan-out.

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
docker compose run --rm index --max-tokens 1024
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
| `RAG_QUEUE_URL` | `--queue-url` | `file://./queue` |
| `RAG_PARSE_QUEUE` | `--parse-queue` | `to-parse` |
| `RAG_INDEX_QUEUE` | `--index-queue` | `to-index` |

The last three are read only by the workers and the producer; the two batch
steps ignore them.

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
python step2_index.py --max-tokens 1024  # smaller chunks than the profile ships
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

### Chunk size without changing models

`--max-tokens` overrides the named profile's sizing in place, so it needs no
`--profile` alongside it:

```bash
docker compose run --rm index --max-tokens 1024
python step2_index.py --max-tokens 1024
```

BGE-M3 accepts 8192 tokens, which is a ceiling rather than a recommendation —
one embedding has to represent everything inside it, and a chunk that spans four
topics retrieves worse than four chunks that each span one. 1024 (minus the
default 128 of headroom, so 896 for the splitter) is the usual starting point
when hits come back topically right but too coarse to quote from.

The model is unchanged, so the vectors stay comparable in dimension — but not in
content, because the text that produced them is different. That makes it a
reprocessing trigger, not a query-time knob: it belongs in the `--chunks-only`
row of the table above, and every document indexed at the old size is now stale.
`profile.version` embeds the number (`BAAI/bge-m3@1024-128`), so `--stale` sees
the change on its own:

```bash
docker compose run --rm index --stale --chunks-only --max-tokens 1024
```

Set it once and leave it. Half a corpus at 8192 and half at 1024 ranks
incoherently — long chunks accumulate loosely-related text that matches many
queries weakly, and they compete against short chunks that match one query
strongly. `RAG_EMBED_MAX_TOKENS` in the environment is the way to keep it
consistent across both hosts and invocations.

---

## Running it in parallel

The batch steps walk a directory. The workers are handed one document at a time
and have no idea how many exist, which is the only difference and the whole
reason a pool of them can run at once without agreeing on anything.

```mermaid
flowchart LR
    producer["<b>enqueue files</b><br><i>S3 event, cron, shell loop</i>"] --> q1[["to-parse"]]

    subgraph pool1 ["parse pool ×N — CPU only, no DB, no model"]
        direction TB
        pw1["parse worker"]
        pw2["parse worker"]
    end

    subgraph pool2 ["index pool ×M — model resident, holds a connection"]
        direction TB
        iw1["index worker"]
        iw2["index worker"]
    end

    q1 --> pool1
    pool1 --> q2[["to-index"]]
    q2 --> pool2
    pool2 --> db[(postgres)]

    pool1 -. "write {sha}.json" .-> store[("shared parse cache<br><i>volume today, S3 later</i>")]
    store -. "load {sha}.json" .-> pool2
```

`N ≫ M`. Parse workers are CPU and nothing else, so they scale with the
backlog; index workers each hold an embedding model and a database connection,
so there are fewer of them and `to-index` is where the difference in throughput
is allowed to pile up. That backlog is the design, not a problem to fix.

```bash
docker compose up -d db
docker compose run --rm enqueue files /data/inbox     # one message per document
docker compose up --scale parse-worker=4 parse-worker index-worker
```

The producer also re-enqueues work that is already parsed — both are whole-cache
or whole-table scans, which is exactly why they belong to a coordinator and not
to thirty replicas:

```bash
docker compose run --rm enqueue cached                # re-index everything cached
docker compose run --rm enqueue stale                 # rechunk after a profile change
```

Workers exit when their queue has been quiet for `--idle-timeout` seconds, which
is what makes `compose up` terminate. Omit it and they run forever, which is
what a Deployment wants.

### The two seams

Nothing above them names a backend, so moving to a cluster is configuration:

| Seam | Today | In a cluster |
|---|---|---|
| `RAG_QUEUE_URL` | `file:///queue` — a directory on a shared volume | `sqs://`, `amqp://`, Redis |
| `RAG_CACHE_DIR` | a bind mount | `s3://bucket/prefix` |

A new queue backend is a `Queue` subclass implementing six transport methods
and one branch in `open_queue()`. Everything above transport — the receive
loop, acking, returning failures, counting attempts, dead-lettering — is
written once in `queues/base.py` and inherited. That is deliberate: swapping
SQS in should not be an opportunity to get the retry semantics subtly
different, and `tests/test_queues.py` runs the same contract against every
backend so a new one either passes or is not done.

A new blob backend is a `BlobStore` with five primitives and two context
managers. The interface yields *paths* rather than bytes because Docling's
`save_as_json` / `load_from_json` want a real file; locally that path is the
cache file itself and nothing is copied, while a remote backend stages a temp
file and transfers around the yield.

### Why workers and not a container per document

`Embedder.__init__` loads a multi-gigabyte model. Amortised over a pod's
lifetime that is a startup cost; paid per document it *is* the pipeline. So the
model and the connection are built before the loop and the loop holds nothing —
`tests/test_wiring.py` asserts the model is loaded once per worker rather than
once per message, because that is the property the whole design rests on and it
would regress silently.

The same argument does not apply to step 1, whose models are much smaller. A
Job per document is defensible there, and is a reasonable escape hatch for
outliers: one 900-page scan that needs 16 GB should not size every parse pod.

### Delivery guarantees

At-least-once, never exactly-once, and the pipeline was already built for it:
work is keyed on the content hash, `documents` is upserted on `sha256`, and both
branches are delete-then-insert scoped to one `document_id`. A redelivered
document converges on the rows it already had.

Two workers handed the *same* document are correct but wasteful — the upsert
takes a row lock, so the second waits out the first and then redoes the work.
Fine for occasional redelivery; worth fixing in the producer if it is routine.

A worker killed mid-document leaves its claim behind, and the next `receive` on
any worker reclaims claims older than `--visibility-timeout` and puts them
back. A document that reliably kills the parser is a poison message: after
`--max-attempts` deliveries it goes to the queue's `dead/` slot and the worker
carries on with the backlog rather than dying with it.

### What to watch when this moves to k8s or ECS

- **Scale on queue depth**, which is what `Queue.depth()` is for — a KEDA
  `ScaledObject` per queue, or SQS backlog-per-task on ECS. Scale the two pools
  separately or the expensive one dictates the cheap one.
- **Pin the thread count.** torch takes every core it can see. Eight parse pods
  on a sixteen-core node without `OMP_NUM_THREADS` set to each pod's CPU limit
  is eight-way oversubscription, and throughput lands *below* serial. Compose
  sets it; a manifest must too.
- **Do not let a cold pod download 2.6 GB** of weights — it defeats the
  autoscaling it is supposed to serve. Build with `PREFETCH_MODELS=1`. That in
  turn weakens the one-image argument: with weights baked, the parse image does
  not need BGE-M3 and the index image does not need the layout models, though it
  still needs the docling library for the chunker.
- **Bound the index pool by Postgres, not by CPU.** Every replica holds a
  connection, and the HNSW index on `chunks.embedding` is a shared write
  structure that stops rewarding parallelism well before the connection limit
  does. Put a pooler in front, and for a large backfill drop the index, load,
  and rebuild.
- **`FileQueue` is the development backend.** It scans `ready/` on every
  receive, so a very deep queue gets slow, and its claim is atomic only where
  `rename` is — fine on a local volume or EBS, not on NFS with the wrong mount
  options.

---

## Checking that the embeddings landed

Step 2 succeeding is not the same as retrieval working. The failures worth
catching here are quiet ones: rows written with a null vector, every vector
identical, or a query embedded through a different profile than the passages
were. Each has its own query, cheapest first.

```bash
docker compose exec db psql -U postgres -d docs
```

**Did anything write.**

```sql
select d.id, d.uri, d.status,
       count(c.id)                              as chunks,
       count(c.embedding)                       as embedded,
       count(*) filter (where c.embedding is null) as missing
from documents d
left join chunks c on c.document_id = d.id
group by d.id order by d.id;
```

`chunks = 0` across the board means step 2 never ran, or ran `--tables-only`.
`missing > 0` is worse: the chunk branch wrote rows but the model produced no
vector for them, so those chunks are unreachable by search while looking present
in every count you take.

**Are the vectors real.**

```sql
select embed_model,
       chunk_config,
       count(*),
       min(sqrt(-(embedding <#> embedding)))::numeric(6,4) as min_norm,
       max(sqrt(-(embedding <#> embedding)))::numeric(6,4) as max_norm,
       count(distinct embedding::text)                     as distinct_vecs
from chunks
group by embed_model, chunk_config;
```

`<#>` is negative inner product, so `sqrt(-(v <#> v))` is the L2 norm. That
detour exists because `l2_norm()` is ambiguous against pgvector's `halfvec` and
`sparsevec` overloads on pg17 — it fails with `function l2_norm(vector) is not
unique`, and an explicit cast does not rescue it.

Norms should be 1.0000 on every row — `encode_passages` normalizes, so anything
else means the vector arrived from some path other than that one. `distinct_vecs`
close to `count` is the check that matters: collapse to a handful means the model
returned the same vector for every input, which is what empty or whitespace chunk
text produces. More than one `chunk_config` group is a half-reprocessed corpus,
the incoherent state the previous section warns about.

**Does similarity behave.** This proves retrieval end to end without loading the
model, by using a stored chunk as its own query:

```sql
with probe as (select id, embedding from chunks order by id limit 1)
select c.id, c.ord,
       (c.embedding <=> p.embedding)::numeric(8,5) as dist,
       left(c.text, 80)                            as snippet
from chunks c, probe p
order by c.embedding <=> p.embedding
limit 10;
```

Row one must be the probe itself at distance ~0; that is the assertion. After it,
distances should rise smoothly over text that is plausibly related. Everything at
~0, or everything bunched at ~1, means degenerate vectors regardless of what the
norm check said.

**Is the index live.**

```sql
explain analyze
select id from chunks
order by embedding <=> (select embedding from chunks limit 1)
limit 10;
```

Look for `Index Scan using chunks_embedding_idx`. A `Seq Scan` on a few thousand
rows is the planner being correct — brute force really is faster there — but on a
full corpus it means the HNSW index isn't being used and every search is scanning
the table.

**Overflow, after any `--max-tokens` change.**

```sql
select document_id, ord, token_count from chunks
where token_count > 8192;                -- or the limit you indexed at
```

**The one that needs the model.** Everything above passes even when the query
side and the passage side disagree about the profile, because nothing above
embeds a query. A mismatch there — wrong model, missing E5 prefix — does not
raise; it just ranks badly:

```python
from rag_embeddings import Embedder, Settings, connect

s = Settings.from_env()
emb, conn = Embedder(s.profile), connect(s.dsn)

qvec = emb.encode_query("what were the Q3 logistics costs?")
for row in conn.execute(
    "select ord, page, left(text, 100) from chunks "
    "order by embedding <=> %s::vector limit 5",
    (qvec,),
).fetchall():
    print(row)
```

Run it with a question you already know the answer to. It is the only check that
covers the query path, and a plausible-looking ranking of the wrong passages is
exactly what a profile mismatch looks like.

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
python tests/test_queues.py
```

`test_wiring.py` stubs Docling, sentence-transformers and psycopg, then runs
step 1 into a temp cache and step 2 out of it, asserting the manifest
round-trip, the statement order inside the transaction, and that
`--tables-only` never loads a model. It then runs the same work over a queue:
producer, two parse workers sharing one queue, one index worker — asserting
that no document is lost or handled twice, that the manifest arrives on the
message rather than being read off disk, that a missing parse is dead-lettered
instead of retried forever while the document behind it still gets written, and
that the model is loaded once per worker rather than once per message.

`test_queues.py` runs one set of assertions against every backend, because that
is the claim the abstraction makes. It covers FIFO order, claim exclusivity,
nack redelivery, attempt counting, visibility-timeout recovery from a worker
that died holding a claim, and dead-lettering — including six threads consuming
one queue with no coordination, which is the property the whole fan-out rests
on.

No torch, no database, no broker. Both run in about a second.

---

## Extending

The obvious next stage is the extraction pass — a schema-driven read over the
chunks (LangExtract, Instructor) producing typed rows in a `facts` table, plus
the entity resolution described in `entity-resolution-design.md`. It belongs as a
third step reading the same cache, for the same reason step 2 does: a change to
your extraction schema shouldn't cost a re-parse.
