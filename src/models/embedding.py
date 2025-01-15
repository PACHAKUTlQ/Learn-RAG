from config import config
from typing import List
import asyncio
from openai import AsyncOpenAI


class OpenAIEmbeddingModel:
    """Wrapper for OpenAI embedding model using official OpenAI package."""

    def __init__(self):
        self.client = AsyncOpenAI(base_url=config.OPENAI_API_BASE,
                                  api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with rate limiting."""
        async with self.semaphore:
            response = await self.client.embeddings.create(input=text,
                                                           model=self.model)
            return response.data[0].embedding

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously."""
        tasks = [self._embed_single(doc) for doc in documents]
        return await asyncio.gather(*tasks)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return await self._embed_single(query)
