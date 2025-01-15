INSERT INTO documents (filepath, filename)
VALUES (%s, %s)
ON CONFLICT (filepath) DO UPDATE
SET updated_at = CURRENT_TIMESTAMP
RETURNING id