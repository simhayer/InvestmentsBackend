import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

_client = None

def get_openai_client() -> OpenAI:
    """
    Singleton OpenAI client.
    Reused across the app to avoid re-initializing per request.
    """
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        _client = OpenAI(api_key=api_key)

    return _client

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
llm_mini = ChatOpenAI(model=os.getenv("OPENAI_MODEL_MINI", "gpt-4o-mini"), temperature=0)