SELECT c.chunk_text, c.metadata, 1 - (e.embedding <=> %s::vector) AS similarity
FROM embeddings e
JOIN chunks c ON e.chunk_id = c.id
ORDER BY e.embedding <=> %s::vector
LIMIT %s