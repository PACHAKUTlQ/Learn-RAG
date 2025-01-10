import chromadb
from config import Config
from uuid import uuid4
from typing import List, Tuple
from datetime import datetime
from langchain_core.documents import Document


class RAGPipeline:
    """RAG Pipeline using Chroma as vector store."""

    def __init__(self,
                 embedding_model,
                 persist_directory: str = Config.CHROMA_PERSIST_DIR,
                 collection_name: str = Config.CHROMA_COLLECTION_NAME):
        """
        Args:
            embedding_model: Embedding model to use
            persist_directory: Directory to persist Chroma data
            collection_name: Name of the Chroma collection
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Configure Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection without embedding function since we provide our own embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "created": datetime.now().isoformat(),
                "description": "RAG collection for document retrieval",
                "embedding_model": "openai"
            })

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the collection with pre-computed embeddings.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        # Prepare data for Chroma with unique UUIDs
        ids = [str(uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = []
        for doc in documents:
            metadata = doc.metadata.copy()
            metadatas.append(metadata)

        # Get embeddings from OpenAI model
        embeddings = self.embedding_model.embed_documents(texts)

        # Add to collection with pre-computed embeddings
        self.collection.add(ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=texts)
        # self.client.persist()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        # Get embedding for the query
        query_embedding = self.embedding_model.embed_query(query)

        # Query the collection using the pre-computed embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"])

        # Convert results to Document objects with scores
        retrieved_docs = []
        for doc_text, metadata, distance in zip(results["documents"][0],
                                                results["metadatas"][0],
                                                results["distances"][0]):
            doc = Document(page_content=doc_text, metadata=metadata)
            # Convert distance to similarity score
            similarity_score = 1.0 - distance
            retrieved_docs.append((doc, similarity_score))

        return retrieved_docs

    def close(self):
        """Explicitly close and persist resources."""
        if hasattr(self, 'client') and self.client is not None:
            # self.client.persist()
            self.client = None

    def __del__(self):
        """Clean up resources during garbage collection."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass
