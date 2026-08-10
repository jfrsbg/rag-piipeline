"""Every statement the pipeline issues, in one place."""

DOC_UPSERT = """
insert into documents (sha256, uri, mime, parser_version, parsed_at, status)
values (%(sha256)s, %(uri)s, %(mime)s, %(parser_version)s, now(), 'ok')
on conflict (sha256) do update set uri = excluded.uri, parsed_at = now()
returning id
"""

TABLE_INSERT = """
insert into doc_tables (document_id, table_index, self_ref, page, caption,
                        num_rows, num_cols, columns, cells, markdown, parser_version)
values (%(document_id)s, %(table_index)s, %(self_ref)s, %(page)s, %(caption)s,
        %(num_rows)s, %(num_cols)s, %(columns)s, %(cells)s, %(markdown)s,
        %(parser_version)s)
"""

CHUNK_INSERT = """
insert into chunks (document_id, ord, text, heading_path, page, refs,
                    embedding, embed_model, chunk_config, token_count)
values (%(document_id)s, %(ord)s, %(text)s, %(heading_path)s, %(page)s, %(refs)s,
        %(embedding)s, %(embed_model)s, %(chunk_config)s, %(token_count)s)
"""

TABLE_DELETE = "delete from doc_tables where document_id = %s"

CHUNK_DELETE = "delete from chunks where document_id = %s"

DOC_ID_BY_SHA = "select id from documents where sha256 = %s"

DOC_EXISTS = "select 1 from documents where sha256 = %s"

STALE_BY_CHUNK_CONFIG = """
select distinct d.sha256 from documents d
join chunks c on c.document_id = d.id
where c.chunk_config <> %s
"""

# --------------------------------------------------------------- read side

# Cosine distance, which is what chunks_embedding_idx is built for: any other
# operator here silently drops to a sequential scan. `chunk_config` comes back
# on every row because a hit embedded through a different profile than the
# query ranks badly rather than failing, and that is the only place it shows.
CHUNK_SEARCH = """
select c.id, c.document_id, c.ord, c.page, c.text, c.heading_path,
       c.chunk_config, d.uri,
       c.embedding <=> %(qvec)s::vector as dist
from chunks c
join documents d on d.id = c.document_id
order by c.embedding <=> %(qvec)s::vector
limit %(limit)s
"""

# Chunks are stored without overlap, so the neighbours are a separate lookup —
# that is what `ord` and chunks_document_ord_idx are for.
CHUNK_WINDOW = """
select ord, text from chunks
where document_id = %(document_id)s
  and ord between %(ord)s - %(window)s and %(ord)s + %(window)s
order by ord
"""

CHUNK_CONFIGS = "select distinct chunk_config from chunks"
