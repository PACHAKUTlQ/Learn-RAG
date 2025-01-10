import os


class Config:
    """Configuration for Chroma and OpenAI."""

    # Chroma settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME",
                                       "rag_collection")
    CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
    CHROMA_SERVER_PORT = os.getenv("CHROMA_SERVER_PORT", "8000")

    # OpenAI settings
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    OPENAI_CHUNK_SIZE = int(os.getenv("OPENAI_CHUNK_SIZE", 300))
