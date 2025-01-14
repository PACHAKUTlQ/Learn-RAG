-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filepath TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table
CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_documents_filepath ON documents(filepath);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 15);

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Stores metadata about original documents';
COMMENT ON TABLE chunks IS 'Stores text chunks and their metadata';
COMMENT ON TABLE embeddings IS 'Stores vector embeddings for each chunk';

COMMENT ON COLUMN documents.filepath IS 'Full path to the document file';
COMMENT ON COLUMN documents.filename IS 'Name of the document file';
COMMENT ON COLUMN chunks.chunk_text IS 'The actual text content of the chunk';
COMMENT ON COLUMN chunks.chunk_index IS 'Position of chunk in document (0-based)';
COMMENT ON COLUMN chunks.start_char IS 'Starting character position in original document';
COMMENT ON COLUMN chunks.end_char IS 'Ending character position in original document';
COMMENT ON COLUMN chunks.metadata IS 'Additional metadata as key-value pairs';
COMMENT ON COLUMN embeddings.embedding IS '1536-dimensional vector embedding of the chunk';