import os
import openai
from langchain.embeddings import OpenAIEmbeddings

class DocumentEmbedder:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = "text-embedding-ada-002"
        openai.api_key = self.api_key
        openai.api_base = self.base_url

    def embed_chunks(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = openai.Embedding.create(
                model=self.model,
                input=chunk
            )
            embeddings.append(response['data'][0]['embedding'])
        return embeddings
