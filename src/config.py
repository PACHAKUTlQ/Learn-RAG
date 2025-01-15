import os
from dotenv import load_dotenv


class Config:
    """Configuration for PostgreSQL and embedding."""

    def __init__(self):
        """Initialize configuration settings, reading from environment variables."""
        self._load_environment_variables()

        # # Chroma settings
        # CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")
        # CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME",
        #                                    "rag_collection")
        # CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
        # CHROMA_SERVER_PORT = os.getenv("CHROMA_SERVER_PORT", "8000")

        # PostgreSQL settings
        self.PG_HOST = os.getenv("PG_HOST", "localhost")
        self.PG_PORT = int(os.getenv("PG_PORT", 5432))
        self.PG_DATABASE = os.getenv("PG_DATABASE", "rag")
        self.PG_USER = os.getenv("PG_USER", "postgres")
        self.PG_PASSWORD = os.getenv("PG_PASSWORD", "")

        # OpenAI settings
        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
        self.OPENAI_CHUNK_SIZE = int(os.getenv("OPENAI_CHUNK_SIZE", 300))

    def _load_environment_variables(self):
        """Loads and validates required environment variables."""
        load_dotenv()

        required_variables = [
            "OPENAI_API_BASE", "OPENAI_API_KEY", "PG_DATABASE", "PG_USER",
            "PG_PASSWORD"
        ]

        missing_vars = [
            var for var in required_variables if not os.getenv(var)
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )


# Singleton instance
config = Config()
