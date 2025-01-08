import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

class DocumentRetriever:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = "text-embedding-ada-002"
        openai.api_key = self.api_key
        openai.api_base = self.base_url
        self.embeddings = OpenAIEmbeddings(model=self.model)

    def retrieve_documents(self, query, embeddings):
        query_embedding = self.embeddings.embed_query(query)
        vector_store = FAISS(embeddings)
        results = vector_store.similarity_search(query_embedding, k=5)
        return results
