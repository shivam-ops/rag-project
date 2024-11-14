import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure OpenAI Configuration
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    # Your Azure OpenAI deployment names
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
    AZURE_EMBEDDINGS_DEPLOYMENT = os.getenv(
        "AZURE_EMBEDDINGS_DEPLOYMENT")

    # Other configurations
    DATA_PATH = "data/test_content.md"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 2