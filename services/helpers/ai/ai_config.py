import os
import openai

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_BASE_URL = os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai")
PPLX_MODEL = os.getenv("PPLX_MODEL", "sonar-pro")  # e.g., sonar-pro / sonar-reasoning-pro
SEARCH_RECENCY = os.getenv("PPLX_SEARCH_RECENCY", "month")  # day|week|month|year

if not PPLX_API_KEY:
    raise RuntimeError("PPLX_API_KEY is not set")

_client = openai.OpenAI(api_key=PPLX_API_KEY, base_url=PPLX_BASE_URL)