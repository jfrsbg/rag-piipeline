-- Applied automatically by the `db` service on first start (see compose).
-- vector(1024) matches BGE-M3; changing the model means dropping and
-- rebuilding the column, not altering it.

create extension if not exists vector;

create table if not exists documents (
  id             bigserial primary key,
  sha256         text unique not null,
  uri            text not null,
  mime           text,
  parser_version text,
  parsed_at      timestamptz,
  status         text default 'pending'
);

create table if not exists doc_tables (
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

create table if not exists chunks (
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

create index if not exists chunks_document_ord_idx
  on chunks (document_id, ord);          -- windowed reads
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists doc_tables_document_idx
  on doc_tables (document_id);
