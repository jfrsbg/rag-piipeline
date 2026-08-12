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

That split is also the fan-out boundary, and it is the only shape the pipeline
has: each step is a pool of services consuming a queue, one document per
message. Nothing walks a directory and runs to completion — see
[Running it in parallel](#running-it-in-parallel).

---

## Layout

```
serve.py                     entrypoint: the search API (FastAPI)
rag_embeddings/
  config.py                  env -> Settings, the only place os.environ is read
  profiles.py                EmbedProfile + the named profiles
  embedder.py                tokenizer, chunker and model, from one profile
  blobstore.py               where the cache lives: local now, object store later
  cache.py                   parse cache and the manifest sidecar
  cli.py                     flags shared by the producer and both services
  shutdown.py                SIGTERM -> finish the document in flight, then stop
  extraction/
    tables.py                branch A: tables -> relational rows
    chunks.py                branch B: chunks -> vectors
  storage/
    connection.py            connect() + register_vector
    sql.py                   every statement, in one place
    writer.py                write_all(): both branches, one commit
  steps/
    parse.py                 what a source string means, for the producer
    search.py                the read side: embed a query, rank chunks
  api/
    app.py                   the service: model and pool, built once at startup
    routes.py                POST /search
    schemas.py               the wire contract
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
tests/test_wiring.py         producer + both services, no torch, no database
tests/test_queues.py         the queue contract, run against every backend
```

The producer and the two services have no entrypoint files: each is run as a
module — `python -m rag_embeddings.workers.parse_worker` — so the container
invocation and the importable `main()` are the same object and cannot drift
apart. `pyproject.toml` also declares them as `rag-enqueue`,
`rag-parse-worker` and `rag-index-worker` for a machine that has the package
installed.

There is no `steps/index.py` and no batch driver in `steps/parse.py`. What is
left of the latter is source resolution — turning `inbox/` or `'*.pdf'` into a
list of files — which is the producer's job, not a consumer's: a service is
handed one uri and never enumerates. It is stdlib-only, and so is everything
else `enqueue` touches: `cache.py` defers its `DocumentConverter` import into
the two functions that parse or load a document, and `workers/__init__.py`
resolves its exports lazily so importing the producer does not import
`index_worker` and, through it, torch.

That is worth a number. Importing `workers.enqueue` costs **0.04s and 131
modules**; eagerly it was **4.0s and 4513**, the whole of docling and torch, paid
on the smallest and most-replicated container in the fan-out to put a filename
on a queue. If you add an import to `cache.py`, `queues/`, or either package
`__init__`, check it against that.

`rag_embeddings/__init__.py` resolves its exports the same way, and for the same
reason: importing `rag_embeddings.queues` is all a producer needs to put a
filename on a queue.

---

## Run it with Docker

```bash
docker compose up -d                         # db, both services, api on :8000
docker compose run --rm enqueue files /data/inbox
```

That is the whole loop. `up` brings the pipeline online and it stays online,
blocking on an empty queue; `enqueue` publishes one message per document and
exits. A drained queue is an idle service, not a finished job, so there is
nothing to re-run when more documents arrive — enqueue them and the pool that
is already up picks them up. See [Querying](#querying) for the read side.

`enqueue` is the one thing here that is a job rather than a service, so it sits
behind a compose profile and `up` does not start it. `compose run` ignores
profiles, which is why the command above works as written.

`./inbox` is mounted read-only at `/data/inbox` and `./cache` holds the parse
cache, so every container sees the same files you do. Model weights land in a
named volume — a rebuild does not re-download BGE-M3.

Arguments after the service name go to the producer:

```bash
docker compose run --rm enqueue files /data/inbox --pattern '*.pdf' --force
docker compose run --rm enqueue cached --tables-only 9f2a...c1
docker compose run --rm enqueue stale
docker compose logs -f parse-worker          # watch the backlog drain
```

Scaling is a flag, and stopping is graceful — SIGTERM lets the document in
flight finish and be acked before the container exits:

```bash
docker compose up -d --scale parse-worker=4
docker compose stop parse-worker
```

Without compose:

```bash
docker build -t rag-embeddings .
docker run --rm -v queue:/queue -v ./inbox:/data/inbox:ro \
  rag-embeddings -m rag_embeddings.workers.enqueue files /data/inbox
docker run -d -v ./cache:/cache -v queue:/queue -v ./inbox:/data/inbox:ro \
  rag-embeddings -m rag_embeddings.workers.parse_worker
docker run -d -v ./cache:/cache -v queue:/queue -v hf-models:/models \
  -e RAG_DSN=postgresql://user:pass@host/docs \
  rag-embeddings -m rag_embeddings.workers.index_worker
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
uv run python -m rag_embeddings.workers.enqueue files inbox/
uv run python -m rag_embeddings.workers.parse_worker &
uv run python -m rag_embeddings.workers.index_worker --dsn postgresql://localhost/docs &
```

Locally the queue is `./queue` (`RAG_QUEUE_URL` defaults to `file://./queue`),
so the two services find each other with no broker to install.

To drain and stop instead of leaving them running — which is what you want for
a one-off backfill or in CI — give them an idle timeout:

```bash
RAG_IDLE_TIMEOUT=5 uv run python -m rag_embeddings.workers.parse_worker
RAG_IDLE_TIMEOUT=5 uv run python -m rag_embeddings.workers.index_worker --dsn postgresql://localhost/docs
```

That is the only thing that makes a service exit on its own. Ctrl-C is the
other way out: the first one finishes the document in flight and exits cleanly,
a second one is taken literally.

First run downloads the Docling layout models and the embedding model (~2 GB for
BGE-M3). CPU works; a GPU makes the initial backfill hours instead of days.

`uv sync` also puts `rag-enqueue`, `rag-parse-worker`, `rag-index-worker` and
`rag-serve` in `.venv/bin` — the same `main()` functions the `-m` invocations
above reach, for a shell where typing the module path is the tedious part. They
need the package importable from the environment rather than from the working
directory, which is the one way they can fail where `-m` does not.

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
| `RAG_EMBED_TOKEN_BUDGET` | `--embed-token-budget` | `16384` |
| `RAG_QUEUE_URL` | `--queue-url` | `file://./queue` |
| `RAG_PARSE_QUEUE` | `--parse-queue` | `to-parse` |
| `RAG_INDEX_QUEUE` | `--index-queue` | `to-index` |
| `RAG_IDLE_TIMEOUT` | `--idle-timeout` | unset — never exit |

The queue variables are read by the producer and both services; the API opens
no queue. `RAG_IDLE_TIMEOUT` is per-container rather than per-deployment, which
is why it is not on `Settings`: leaving it unset is what makes a service block
on an empty queue instead of treating a drained backlog as a finish line.

`RAG_EMBED_TOKEN_BUDGET` is the odd one out: it is hardware sizing, not model
semantics. It caps tokens per forward pass — sequences times padded length —
rather than sequences per batch, because that product is what a transformer's
peak allocation actually follows. It changes throughput and memory only; the
vectors and `profile.version` are identical either way, so it never makes a row
stale. Lower it if `index-worker` is OOM-killed (exit 137); see
[Things that will bite you](#things-that-will-bite-you).

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

The work is published, not invoked. `enqueue` decides which files exist and puts
one message on `to-parse` per document; whichever `parse-worker` replica is free
claims it:

```bash
python -m rag_embeddings.workers.enqueue files inbox/                      # a directory
python -m rag_embeddings.workers.enqueue files inbox/ --pattern '*.pdf'    # filtered
python -m rag_embeddings.workers.enqueue files a.pdf b.pdf 'reports/*.docx'
python -m rag_embeddings.workers.enqueue files inbox/ --force              # after a Docling upgrade
```

Source resolution lives on the producer side deliberately: a directory walk is
an enumeration, and a pool of thirty replicas each enumerating the same prefix
is a scan, not a pipeline. A consumer is handed one uri and never asks how much
work exists.

Each service writes two files per document: `{sha}.json` (the parse) and
`{sha}.meta.json` (uri, mime, parser version, timestamp), then announces the sha
on `to-index` — after the parse is stored, never before, because the downstream
consumer may be on another node and will look for it immediately.

The manifest exists because the steps are separate processes, usually on
separate machines. Step 2 only gets a sha; rather than have it re-derive the uri
from a source that may no longer be reachable, step 1 records what it knew at
parse time. It also rides on the
message, so step 2 normally never reads the sidecar at all.

Content-addressed and idempotent: a redelivered message over the same bytes is a
cache hit, which is what makes at-least-once delivery safe here — no duplicates,
no cleanup.

For object storage, stage the bytes to disk and keep the real location:

```bash
aws s3 sync s3://bucket/prefix ./inbox
python -m rag_embeddings.workers.enqueue files inbox/ --uri-prefix s3://bucket/prefix
```

That is what ends up in `documents.uri`, while Docling opens the local file. In
production `enqueue files` is replaced by whatever watches the bucket — an S3
event, a Lambda, a cron over a prefix. It publishes the same message, which is
why the services do not care which one you use.

---

## Step 2 — extract and store

Also published rather than invoked. The parse service announces new documents
automatically; the commands below are for re-indexing what is already cached:

```bash
python -m rag_embeddings.workers.enqueue cached                 # everything in the cache
python -m rag_embeddings.workers.enqueue cached 9f2a...c1 4b81...0e
python -m rag_embeddings.workers.enqueue stale                  # only what the profile made stale
```

Each `index-worker` loads a cached parse, runs both branches, and writes them in
one transaction per document. No parser runs here, and the embedding model is
loaded once per container rather than once per document — that amortisation is
the whole reason this is a service.

Which branches run is carried on the message, not chosen by a flag on the
process doing the work: a pool has no per-run flags to set. The producer decides,
and these are the reprocessing triggers that make the two branches separate code
paths at all:

| Changed | Run | Re-parses? | Loads a model? |
|---|---|---|---|
| Extraction logic / table schema | `enqueue cached --tables-only` | no | no |
| Embedding model, `max_tokens`, chunker | `enqueue cached --chunks-only` | no | yes |
| Docling version, OCR settings | `enqueue files --force` | yes | yes |

After changing the profile, find exactly what is stale rather than reprocessing
everything:

```bash
python -m rag_embeddings.workers.enqueue stale
```

`stale` compares each chunk's stored `chunk_config` against the active profile
version, so it catches model swaps and token-limit changes alike, and publishes
chunks-only messages — the parse and the tables did not change, and rewriting
identical table rows costs without buying anything.

Note that `--profile` and `--max-tokens` belong on the *service*, not on
`enqueue`: the producer selects documents, the consumer decides how they are
embedded. Set them in the environment (`RAG_EMBED_PROFILE`,
`RAG_EMBED_MAX_TOKENS`) so the API and `index-worker` cannot disagree.

### Chunk size without changing models

`--max-tokens` overrides the named profile's sizing in place, so it needs no
`--profile` alongside it:

```bash
RAG_EMBED_MAX_TOKENS=1024 docker compose up -d index-worker
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
`profile.version` embeds the number (`BAAI/bge-m3@1024-128`), so `stale` sees
the change on its own:

```bash
RAG_EMBED_MAX_TOKENS=1024 docker compose up -d index-worker
docker compose run --rm enqueue stale
```

Set it once and leave it. Half a corpus at 8192 and half at 1024 ranks
incoherently — long chunks accumulate loosely-related text that matches many
queries weakly, and they compete against short chunks that match one query
strongly. `RAG_EMBED_MAX_TOKENS` in the environment is the way to keep it
consistent across both hosts and invocations.

---

## Running it in parallel

This is not a mode — it is how the pipeline runs. Each service is handed one
document at a time and has no idea how many exist, which is the whole reason a
pool of them can run at once without agreeing on anything, and the reason there
is no directory-walking driver left to fall back on.

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
docker compose up -d --scale parse-worker=4
docker compose run --rm enqueue files /data/inbox     # one message per document
```

The producer also re-enqueues work that is already parsed — both are whole-cache
or whole-table scans, which is exactly why they belong to a coordinator and not
to thirty replicas:

```bash
docker compose run --rm enqueue cached                # re-index everything cached
docker compose run --rm enqueue stale                 # rechunk after a profile change
```

Both services block on an empty queue and run until they are stopped, which is
what a Deployment wants: `to-index` sitting empty for an hour is a drained
backlog, not a reason to exit. `--idle-timeout` (or `RAG_IDLE_TIMEOUT`) is the
drain-and-stop shape — a container exits once its queue has been quiet that many
seconds, which is what makes `compose up` terminate on a laptop or in CI:

```bash
RAG_IDLE_TIMEOUT=5 docker compose up parse-worker index-worker
```

Stopping them the normal way is graceful. SIGTERM — `compose stop`, a rolling
update, a scale-down, Ctrl-C — sets a flag rather than ending the process; the
loop finishes the document in flight, acks it, and exits 0, so the restart
policy sees a clean stop and not a crash. A second signal is taken literally.
`stop_grace_period` in compose is set to 120s for both services, which only has
to exceed one document. Without that, SIGKILL lands mid-parse and the claimed
message waits out the visibility timeout before anyone retries it — every deploy
would stall on whatever was in flight.

Note that an idle timeout interacts with `--visibility-timeout` (default 300s):
a worker that exits while a message is still in flight leaves it in `inflight`,
and the next worker to start reclaims it with an attempt already spent. Short
idle timeouts and long visibility timeouts can therefore burn a message's
`--max-attempts` across restarts without it ever being tried properly.

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

That snippet is `steps/search.py`, with the profile check added, served over
HTTP by `serve.py`:

```bash
docker compose up -d api
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "what were the Q3 logistics costs?"}'
```

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

The read side is a service — one query in, ranked chunks out, over HTTP:

```bash
docker compose up -d api                                # stays up
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "what were the Q3 logistics costs?"}'
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "patrimônio líquido em junho", "limit": 10}'
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "logistics costs", "window": 1}'     # neighbours too

python serve.py                                         # or locally
```

Interactive docs, generated from the schemas, are at `localhost:8000/docs`.

It is a service rather than a script for one reason: the model. `Embedder`
loads ~2.1 GB, which a one-shot query paid on every invocation and a process
that stays up pays once, at startup. The database is the same trade in
reverse — a pool of connections borrowed per request, opened before the
service reports itself as up so an unreachable database fails the container
instead of the first query.

It reads the same `RAG_EMBED_PROFILE` / `RAG_DSN` environment as step 2, which
is the point: the query has to be embedded by the model that embedded the
passages. It cannot enforce that, so it checks — a corpus indexed under a
`chunk_config` the active profile doesn't match logs a warning at startup, and
any hit that disagrees comes back with `"stale": true`. Nothing raises, because
a profile mismatch never raises; it just ranks the wrong passages convincingly.

```json
{
  "query": "what were the Q3 logistics costs?",
  "profile": "bge-m3/8192/128",
  "count": 1,
  "hits": [
    {
      "id": 412, "document_id": 7, "ord": 12, "page": 4,
      "uri": "inbox/report.pdf",
      "heading_path": ["Q3", "Costs"],
      "chunk_config": "bge-m3/8192/128",
      "distance": 0.1873, "stale": false,
      "text": "Logistics costs fell ..."
    }
  ]
}
```

Read the distances rather than the top hit alone: they should rise smoothly,
and everything bunched at ~0 or ~1 means degenerate vectors, which is the
[previous section's](#checking-that-the-embeddings-landed) check.

The service has its own image, `Dockerfile.api` — the same dependency layers as
the pipeline image up to the `api` extra, which adds FastAPI and uvicorn and is
what `docker compose up api` builds. The step scripts are deliberately not in
it: nothing that answers requests from outside should be able to start a parse.

`-k` is how many chunks come back, not how many documents; a long report can
own every row. `--window N` prints N chunks either side of each hit, which is
the widening query below done for you — worth it whenever a hit reads as though
it starts mid-sentence, because it does.

Under the hood, embed the query through the same profile — same model, same
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

**One long chunk decides the whole batch's memory.** `index-worker` dying with
exit 137 and no traceback is the OOM killer, not a bug in your document — look
for `OOMKilled: true` in `docker inspect`. A batch is padded to its longest
member, so a single 5 000-token chunk among 500-token ones makes every sequence
in the pass 5 000 tokens wide; peak allocation follows sequences × padded
length × the model's 4096-wide feed-forward layer, which is why a fixed batch
size bounds nothing. `encode_passages` therefore batches to
`RAG_EMBED_TOKEN_BUDGET` tokens rather than to a count of sequences. Lower the
budget before raising `mem_limit` — the compose file caps `index-worker` at 4 GB
so a bad batch fails as one container instead of taking the database down with
it.

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

`test_wiring.py` stubs Docling, sentence-transformers and psycopg, then drives
the real entrypoints from argv — producer, parse service, index service, in that
order against a file-backed queue — so the flags and the console scripts are
covered along with the manifest round-trip, the statement order inside the
transaction, and the fact that `--tables-only` never loads a model.
`--idle-timeout 0` is the only concession to running three services in one
process.

It then runs the same work as a pool: two parse services sharing one queue and
one index service, asserting that no document is lost or handled twice, that the
manifest arrives on the message rather than being read off disk, that a missing
parse is dead-lettered instead of retried forever while the document behind it
still gets written, and that the model is loaded once per container rather than
once per message.

Last it checks shutdown, which is the part a service has and a job does not: a
stop signal arriving mid-backlog must let the document in flight finish and be
acked, leave the rest of the queue untouched, and report a signalled exit rather
than a crash — and SIGTERM must set that flag instead of killing the process.

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
