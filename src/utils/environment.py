import os
from dotenv import load_dotenv


def load_environment_variables():
    load_dotenv()
    required_variables = ["OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_MODEL"]
    for var in required_variables:
        if not os.getenv(var):
            raise ValueError(f"Missing environment variable: {var}")
