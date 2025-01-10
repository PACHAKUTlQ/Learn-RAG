import os
from openai import OpenAI
from dotenv import load_dotenv


class OpenAITextEmbedding:

    def __init__(self):
        load_dotenv()
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL")

    def run(self, text):
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        response = client.embeddings.create(input=text, model=self.model)

        return response.data[0].embedding
