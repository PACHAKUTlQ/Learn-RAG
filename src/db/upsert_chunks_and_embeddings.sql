WITH data AS (
    SELECT x.doc_id,
        x.page_content,
        x.chunk_index,
        x.start_char,
        x.end_char,
        x.metadata_json::jsonb as metadata,
        x.embedding
    FROM (VALUES %s)
    AS x(doc_id, page_content, chunk_index, start_char, end_char, metadata_json, embedding)
),
inserted_chunks AS (
    INSERT INTO chunks
        (document_id, chunk_text, chunk_index, start_char, end_char, metadata)
    SELECT doc_id, page_content, chunk_index, start_char, end_char, metadata
    FROM data
    ON CONFLICT (document_id, chunk_index) 
    DO UPDATE SET 
        chunk_text = EXCLUDED.chunk_text,
        start_char = EXCLUDED.start_char,
        end_char = EXCLUDED.end_char,
        metadata = EXCLUDED.metadata
    RETURNING id, chunk_index
)
INSERT INTO embeddings (chunk_id, embedding)
SELECT ic.id, d.embedding
FROM inserted_chunks ic
JOIN data d ON ic.chunk_index = d.chunk_index
ON CONFLICT (chunk_id)
DO UPDATE SET 
    embedding = EXCLUDED.embedding