import psycopg2
from psycopg2.extras import execute_values
from config import config
from typing import List, Tuple
from langchain_core.documents import Document
import json


class RAGPipeline:
    """RAG Pipeline using PostgreSQL and pgvector as vector store."""

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: Embedding model to use
        """
        self.embedding_model = embedding_model
        self.connection = self._create_connection()

    def _create_connection(self):
        """Create and return a PostgreSQL connection."""
        return psycopg2.connect(dbname=config.PG_DATABASE,
                                user=config.PG_USER,
                                password=config.PG_PASSWORD,
                                host=config.PG_HOST,
                                port=config.PG_PORT)

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with pre-computed embeddings.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        # Prepare data for insertion
        embeddings = await self.embedding_model.embed_documents(
            [doc.page_content for doc in documents])

        with self.connection.cursor() as cur:
            # Insert documents and get their IDs
            doc_ids = []
            for doc in documents:
                try:
                    cur.execute(
                        """
                        INSERT INTO documents (filepath, filename)
                        VALUES (%s, %s)
                        ON CONFLICT (filepath) DO UPDATE
                        SET updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """, (doc.metadata['filepath'], doc.metadata['filename']))
                except Exception as e:
                    print(f"Error inserting document: {e}")
                    continue
                doc_ids.append(cur.fetchone()[0])

            # Insert chunks and embeddings
            chunk_data = []
            for doc_id, doc, embedding in zip(doc_ids, documents, embeddings):
                chunk_data.append(
                    (doc_id, doc.page_content,
                     doc.metadata.get('chunk_index',
                                      0), doc.metadata.get('start_char', 0),
                     doc.metadata.get('end_char',
                                      0), json.dumps(doc.metadata), embedding))

            # Insert chunks and embeddings in a single transaction
            try:
                execute_values(cur,
                               """
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
                    """,
                               chunk_data,
                               page_size=100)

                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                print(f"Error inserting chunks: {e}")

    async def retrieve(self,
                       query: str,
                       k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        query_embedding = await self.embedding_model.embed_query(query)

        with self.connection.cursor() as cur:
            # Set the number of probes for the IVFFlat index
            cur.execute("SET ivfflat.probes = 10")

            try:
                cur.execute(
                    """
                    SELECT c.chunk_text, c.metadata, 1 - (e.embedding <=> %s::vector) AS similarity
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, k))
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                return []

            results = cur.fetchall()

            print(len(results), "results retrieved")

            retrieved_docs = []
            for chunk_text, metadata, similarity in results:
                metadata_dict = metadata if isinstance(
                    metadata, dict) else json.loads(metadata)
                doc = Document(page_content=chunk_text, metadata=metadata_dict)
                retrieved_docs.append((doc, similarity))

            return retrieved_docs

    def close(self):
        """Close database connection."""
        if hasattr(self, 'connection') and self.connection is not None:
            self.connection.close()
            self.connection = None

    def __del__(self):
        """Clean up resources during garbage collection."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass
