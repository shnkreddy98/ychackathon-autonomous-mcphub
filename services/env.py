import os

from dotenv import load_dotenv

load_dotenv("dev.env")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://localhost:3000")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", None)
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", None)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
MONGODB_URI = os.getenv(
    "MONGODB_URI", "mongodb://admin:admin123@localhost:27017/agent_db?authSource=admin"
)

