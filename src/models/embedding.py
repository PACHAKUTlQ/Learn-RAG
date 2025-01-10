from config import Config
from typing import List
from openai import OpenAI


class OpenAIEmbeddingModel:
    """Wrapper for OpenAI embedding model using official OpenAI package."""

    def __init__(self):
        self.client = OpenAI(base_url=Config.OPENAI_API_BASE,
                             api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        # Batch process documents to handle large lists efficiently
        embeddings = []
        for doc in documents:
            response = self.client.embeddings.create(input=doc,
                                                     model=self.model)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(input=query, model=self.model)
        return response.data[0].embedding
