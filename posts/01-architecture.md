# Building a RAG platform that scales

### Part 1 — The architecture

I've spent the past months sharpening my AI skills, and at some point I realized that reading papers and running notebooks has a ceiling. To actually take this to the next level, I needed an end-to-end project — something with real infrastructure, real failure modes, and real cost decisions attached to it.

So I decided to build a full AI platform from scratch, binding Platform Engineering and AI together in one project. This post is the first part of that journey.

The project is big, and it's meant to be a continuous learning exercise rather than a weekend build. It's not 100% done yet, and I'll keep adding to it until the full architecture is in place. But I can already tell you what you'll find in there once it's finished:

- **AWS infrastructure with Terraform** — S3, SQS, ECS, RDS
- **Docling** for parsing and indexing — caching, chunks and embeddings
- **pg_vector** for vector storage and embedding retrieval
- **FastAPI with the Anthropic SDK** for querying pg_vector and generating augmented answers
- **Data modeling for unstructured data**

This first post walks through the architecture. Not the code — the shape of the thing, why each piece is where it is, and which parts already exist. The following posts in the series will go deep into each layer as I build it.

---

## The problem the architecture is solving

A RAG demo is easy. You load a PDF, chunk it, embed it, stuff the top hits into a prompt, and it works. You can do that with a few lines.

What makes it hard is everything that comes after "it works on my laptop":

- Parsing a PDF is slow and CPU-bound. Embedding is memory-bound and wants a GPU. Those are two completely different machines, and if you put them in the same process you're paying for the wrong hardware on both sides.
- Loading an embedding model costs a couple of gigabytes and several seconds. Pay that once per container and it's a startup cost. Pay it once per document and it is your pipeline.
- Treating documents as plain text loses information. Financial reports are built from tables and figures, and a table has to be flattened into text before anything can embed it. There's no vector for a grid. The mistake isn't flattening; it's flattening *one-way*, so the numbers you can no longer compute against are the only thing you kept.

The architecture below is my answer to those three problems. Everything in it exists because one of them forced it.

---

## The architecture

![Architecture diagram](./rag-pipeline.drawio.svg)

There are two halves, and the seam between them is the most important decision in the whole design.

### Storage layer: turning documents into retrievable data

**Sources → S3.** PDFs, plain text, transcripts, whatever else. Raw bytes land in a bucket. That's the entry point and it's deliberately dumb: no processing, no validation, just landing.

**S3 event → SQS (`to-parse`).** An object landing fires an event that publishes one message per document. This is the producer, and it's the only component in the system that ever *enumerates*. Everything downstream is handed exactly one document and never asks how much work is left. That's what lets a pool of thirty workers run at once without coordinating with each other.

**Parser Pool (ECS).** N workers pulling from `to-parse`. Each one takes a document, runs Docling over it, and writes the parsed result to a S3 bucket: the SHA-256 of the source bytes is the key, and a small sidecar records the URI, MIME type, parser version and timestamp.

That cache is the single most valuable artifact in the pipeline. Parsing is the expensive, deterministic step, the same bytes always produce the same parse. Caching it means that when I change my chunking strategy, or swap embedding models, or fix a bug in table extraction, I re-run only the affected step. I never re-parse.

It's also what makes at-least-once delivery safe. A redelivered message over the same bytes is just a cache hit. No duplicates, no cleanup logic.

**Parser cache → SQS (`to-index`).** After the parse is stored, the worker announces the SHA on the second queue. Ordering matters here, because the consumer is on a different machine and will go looking for that file immediately.

**Index Pool (ECS).** M workers pulling from `to-index`. Each one loads a cached parse and **fans it out to two sinks**:

- **Branch A — tables into relational columns.** Cells, headers, page, caption, the full grid as JSONB. Typed data stays typed.
- **Branch B — everything else into chunk embeddings.** Chunked with Docling's hybrid chunker, contextualized with the heading path, embedded with BGE-M3.

Both branches land in **one transaction per document**. A document is either fully indexed or not indexed at all. There's no state where the tables are current and the vectors are three versions behind.

**RDS Postgres + pg_vector.** Three tables: `documents`, `doc_tables`, `chunks`. Vectors are `vector(1024)` matching BGE-M3, with an HNSW index on cosine distance.

Tables go to *both* sinks on purpose, and this is the answer to the flattening problem from the top of the post. The chunk branch does flatten a table into text — it has to, because that's the only form an embedding model accepts. But it's a lossy *copy*, not a lossy *conversion*: every chunk records a reference to the objects it was built from, and those references are join keys back into `doc_tables`. So a hit on a flattened table resolves straight to the authoritative grid it came from.

Search finds it; the app computes on it. Nothing is destroyed, because nothing was replaced.

### Retrieval layer — turning a question into an answer

**Search query → FastAPI (ECS).** The read side is a long-running service for exactly the same reason the index workers are: the model. Embedding a query needs the *same* model that embedded the passages, and loading it costs ~2 GB. A service pays that once at startup. A script would pay it on every question.

**bge-m3 → pg_vector.** The query is embedded through the identical profile, same model, same normalization, same prefix conventions, and ranked against the corpus by cosine distance.

**Retrieved context → Claude.** The top chunks (optionally widened with their neighbours, since chunks are stored without overlap) become the context for generation via the Anthropic SDK. That's the "augmented" part of retrieval-augmented generation, and it's the piece I'm building next.

---
Review
## The two asymmetries worth noticing

**The pools are not the same size.** `N ≫ M`, and the reason is not the one you'd guess.

The obvious story would be that parsing is the quick step and embedding is the slow one, so the parse pool races ahead. That's wrong, and measuring it says so. Both halves are model inference. Parsing runs a layout model on *every page* and a table-structure model on *every table*, largely one page at a time. Embedding batches thirty-odd chunks into a single forward pass. For a fifty-page PDF that's fifty-plus sequential passes on one side and a handful of batched ones on the other. **Parsing is the slower stage, by a wide margin.**

So the asymmetry was never speed. It's what each worker *holds*, and therefore what caps the number of them.

A parse worker holds nothing anyone else touches: it reads bytes, burns CPU, writes a new key. No shared write structure, no connection to a finite pool. Throughput is linear in worker count, and the only thing that stops me adding another is what I'm willing to pay. It's also safely interruptible — a worker killed mid-document just means the message comes back and the same bytes parse to the same key — so this pool can live on spot capacity.

An index worker holds a resident embedding model and a Postgres connection. Per worker it's CPU-bound like any other, but the *count* is capped by something the pool doesn't own. Connections are finite and scale with the instance class, not with my ambition. And the HNSW index on `chunks.embedding` is a shared write structure that stops rewarding parallelism well before the connection limit does — past that point, more workers don't add throughput, they add contention.

**The cheap pool's ceiling is my budget. The expensive pool's ceiling belongs to Postgres.** That's the whole reason the seam exists. Fused into one process, the unit of scale would be "a thing that holds a 2 GB model and a database connection," and I'd need N of those to keep up with the parse backlog — renting memory-heavy capacity to do work that needs none of it, and handing a database veto over a stage that never talks to it.

**Only one half is cacheable, and both halves know it.** The queue is at-least-once, so every message can arrive twice. The two sides answer that completely differently.

Parsing is deterministic: the same bytes always produce the same parse, so the result is content-addressed and a redelivery is a cache hit. Retry costs nothing and needs no cleanup logic. That's what makes the slow, expensive step safe to run on interruptible hardware — the property that makes it cacheable is the same property that makes it disposable.

Indexing has no such luck. It writes to shared mutable state, and you can't hash your way out of that. Its safety comes from the other direction: one transaction per document, both branches inside it. A document is either fully indexed or not indexed at all.

Same delivery guarantee, two entirely different defences. The step you can re-run for free, you re-run. The step you can't, you make atomic.

---

## What already exists

The storage layer is largely built. The current codebase runs the full pipeline locally under Docker Compose — producer, parse pool, index pool, Postgres with pg_vector, and the search API:

- **Both worker pools**, each scalable with a flag, each consuming one document per message, each shutting down gracefully on SIGTERM by finishing the document in flight before exiting.
- **The parse cache**, content-addressed with its manifest sidecar.
- **The two-sink write**, tables and chunks, in one transaction.
- **The full pg_vector schema** with the HNSW index.
- **The FastAPI search service**, serving ranked chunks over HTTP with optional neighbour windows.
- **Reprocessing triggers** — the pipeline knows which change invalidates which step. Changed the table schema? Re-extract, no re-parse, no model. Changed the embedding model or chunk size? Re-chunk only. Upgraded Docling? Only then does anything re-parse. The profile version is stored on every row, so the system can tell you exactly which documents went stale.
- **A profile-mismatch check**, because querying with a different model than you indexed with doesn't raise an error — it just ranks the wrong passages, convincingly. So the service compares them at startup and flags stale hits in the response.

The two pieces that will become AWS are already abstracted behind interfaces, which is the whole reason the move is a configuration change rather than a rewrite:

| Seam | Today | On AWS |
|---|---|---|
| Queue | A directory on a shared volume | SQS |
| Blob store | A bind mount | S3 |

Everything above transport — the receive loop, acking, returning failures, counting attempts, dead-lettering poison messages — is written once and inherited by every backend. Swapping SQS in shouldn't be an opportunity to get the retry semantics subtly wrong, and there's a shared contract test suite that every backend has to pass.

---

## What's next

Three things, and they map to the next posts in this series:

1. **Infrastructure with Terraform.** S3 buckets and events, both SQS queues with their dead-letter configuration, ECS task definitions and services for the three pools, RDS with pg_vector. Plus the part that actually makes it a *platform*: autoscaling the two pools on different signals. The parse pool scales on `to-parse` depth, because depth there is a real backlog and the answer is always more workers. The index pool can't use its own queue depth — if parsing is the slower stage, `to-index` stays shallow no matter how far behind the indexers are — so it scales on commit latency instead, which is the signal that actually knows where the database's ceiling is.

2. **Data modeling for unstructured data.** This is the part I find most interesting and the most under-discussed in RAG content. A chunk is not just text — it has a position, a heading path, a page, provenance, and references back to the structured objects it was derived from. Getting that model right is what separates "the search returned something plausible" from "the application can act on it."

3. **Agentic retrieval.** FastAPI with the Anthropic SDK, closing the loop from question to grounded answer — and going beyond single-shot retrieval into an agent that can query, widen context, follow a chunk to its source table, and decide when it has enough to answer.

The interesting problems aren't in getting a vector search to return results. They're in everything around it — cost, staleness, failure, scale, and giving the model data that's actually structured enough to reason over.

That's what I'll be writing about next.
