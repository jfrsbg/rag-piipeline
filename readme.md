# pipeline.py

Parses documents once, caches the parse, and fans the result out to two sinks: tables into relational columns, everything else into chunk embeddings. Both writes land in a single transaction.

```
object storage ──► parse ──► cache/parsed/{sha}.json
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              doc.tables                  chunker.chunk()
                    │                           │
                    ▼                           ▼
              doc_tables                  chunks + embeddings
                    └───────── one commit ──────┘
```

---

## Install

```bash
pip install docling docling-core sentence-transformers transformers \
            psycopg[binary] pgvector
```

First run downloads the Docling layout models and the embedding model (~2 GB for BGE-M3). CPU works; a GPU makes the initial backfill hours instead of days.

---

## Database setup

The module assumes these tables exist. Run once:

```sql
create extension if not exists vector;

create table documents (
  id             bigserial primary key,
  sha256         text unique not null,
  uri            text not null,
  mime           text,
  parser_version text,
  parsed_at      timestamptz,
  status         text default 'pending'
);

create table doc_tables (
  id             bigserial primary key,
  document_id    bigint not null references documents(id) on delete cascade,
  table_index    int not null,
  self_ref       text not null,          -- '#/tables/0' — join key to chunks
  page           int,
  caption        text,
  num_rows       int,
  num_cols       int,
  columns        jsonb,                  -- header cell texts
  cells          jsonb,                  -- full grid with row/col offsets
  markdown       text,
  parser_version text,
  unique (document_id, table_index)
);

create table chunks (
  id           bigserial primary key,
  document_id  bigint not null references documents(id) on delete cascade,
  ord          int not null,
  text         text not null,
  heading_path text[],
  page         int,
  refs         jsonb,                    -- self_refs of the items serialized
  embedding    vector(1024),
  embed_model  text not null,
  chunk_config text not null,
  token_count  int,
  unique (document_id, ord)
);

create index on chunks (document_id, ord);          -- windowed reads
create index on chunks using hnsw (embedding vector_cosine_ops);
create index on doc_tables (document_id);
```

`vector(1024)` matches BGE-M3. Change it if you change models — and note that altering the dimension requires dropping and rebuilding the column.

Before building the HNSW index on a large table, raise `maintenance_work_mem` (1–2 GB) or the build will take far longer than it should.

---

## Configuration

Everything model-related lives in one frozen dataclass. The tokenizer, the chunker, and the embedder are all derived from it, so a mismatch between chunk sizing and embedding capacity requires constructing two profiles — which shows up in a diff.

```python
from pipeline import EmbedProfile, Embedder, connect

BGE_M3 = EmbedProfile(model_id="BAAI/bge-m3", max_tokens=8192)

emb  = Embedder(BGE_M3)
conn = connect("postgresql://localhost/docs")
```

For E5-family models, set the prefixes — omitting them degrades retrieval quietly rather than loudly:

```python
E5 = EmbedProfile(
    model_id="intfloat/multilingual-e5-large",
    max_tokens=512,
    passage_prefix="passage: ",
    query_prefix="query: ",
)
```

`max_tokens` comes from the model card, deliberately not from `tokenizer.model_max_length` — many HF configs ship a sentinel in the range of 10^19 there, and feeding that to the chunker means it never splits anything.

`headroom` (default 128) sizes the chunker below the real limit because `contextualize()` prepends the heading path *after* the split decision was made.

Also set `PARSER_VERSION` and `CACHE_DIR` at the top of the module to match your environment.

---

## Ingesting

```python
from pathlib import Path

for path in Path("inbox").glob("*.pdf"):
    sha = ingest(conn, str(path), path.read_bytes(),
                 mime="application/pdf", emb=emb)
```

`ingest()` is idempotent by content hash — re-running over the same file is a no-op, so a crashed batch can be restarted from the top without duplicates or cleanup.

From object storage rather than local disk:

```python
blob = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
sha = ingest(conn, f"s3://{bucket}/{key}", blob, mime="application/pdf", emb=emb)
```

Note that `parse_and_cache` passes `uri` to Docling's converter. For remote objects, write the bytes to a temp file first and pass that path, keeping the s3 URI as the `uri` recorded in `documents`.

---

## Reprocessing

The three entry points exist because the branches have different reprocessing triggers. All three read the same cache; only `ingest` ever touches a parser.

| Changed | Run | Re-parses? |
|---|---|---|
| Extraction logic / table schema | `reextract(conn, sha)` | no |
| Embedding model, `max_tokens`, chunker | `rechunk(conn, sha, emb)` | no |
| Docling version, OCR settings | `ingest(...)` again | yes |

After changing the profile, find exactly what's stale rather than reprocessing everything:

```python
emb = Embedder(EmbedProfile("BAAI/bge-m3", max_tokens=2048))   # was 8192

for sha in stale_documents(conn, emb):
    rechunk(conn, sha, emb)
```

`stale_documents()` compares each chunk's stored `chunk_config` against the active profile version, so it catches model swaps and token-limit changes alike.

To force a re-parse (Docling upgrade, better table model), delete the cache entries and re-ingest:

```bash
rm cache/parsed/{sha}.json
```

---

## Querying

**Semantic search.** Embed the query through the same profile — same model, same normalization, same prefix:

```python
qvec = emb.encode_query("what were the Q3 logistics costs?")

rows = conn.execute(
    """select id, document_id, ord, text, page
       from chunks order by embedding <=> %s::vector limit 20""",
    (qvec,),
).fetchall()
```

**Widen the context after retrieval.** Chunks are stored without overlap; the neighbours are one indexed lookup away, which is why `ord` exists:

```python
window = conn.execute(
    """select text from chunks
       where document_id = %s and ord between %s - 1 and %s + 1
       order by ord""",
    (document_id, ord_, ord_),
).fetchall()
```

**Follow a table chunk to its authoritative cells.** A hit whose `refs` contains a table's `self_ref` resolves to the exact grid:

```sql
select t.cells, t.columns, t.page
from chunks c
join doc_tables t
  on t.document_id = c.document_id
 and c.refs @> to_jsonb(t.self_ref)
where c.id = %s;
```

That's the point of sending tables to both sinks — search finds the readable markdown copy, the application reads the typed values.

---

## Things that will bite you

**Overflow warnings are real.** `build_chunks` logs `chunk overflow` when the contextualized text exceeds the profile limit. It does not truncate or split — it records `token_count` so you can find them later:

```sql
select document_id, ord, token_count from chunks where token_count > 8192;
```

If that returns rows, either raise `headroom` or add a hard splitter for those cases. Wide tables and deep heading paths are the usual causes.

**`ord` must stay dense.** It's assigned by `enumerate` over the chunker's output, and the emission order *is* reading order. If you add filtering (dropping empty or boilerplate chunks), filter after assignment or the windowed reads above will silently return short.

**Table cells, not DataFrames.** `extract_tables` reads `table.data.table_cells` because `export_to_dataframe()` has a known class of bug where a column vanishes from the export while the data sits intact in the JSON. If you need a frame for analysis, build it from `cells` yourself.

**`prov` can be empty.** Merged list groups and some HTML-derived nodes carry no provenance, so `page` will be `null` for them. That's expected, not a failure — don't add a NOT NULL constraint there.

**Both branches delete before insert.** `write_all`, `reextract`, and `rechunk` each clear the document's existing rows first, inside the transaction. Reprocessing is therefore a replace, not an append.

---

## Extending

The obvious next stage is the extraction pass — a schema-driven read over the chunks (LangExtract, Instructor) producing typed rows in a `facts` table, plus the entity resolution described in `entity-resolution-design.md`. Both belong as a fourth entry point reading the same cache, for the same reason the other three do: a change to your extraction schema shouldn't cost a re-parse.