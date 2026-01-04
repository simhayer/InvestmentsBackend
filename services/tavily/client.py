from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Dict

import httpx

TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_TIMEOUT_SEC = float(os.getenv("TAVILY_TIMEOUT_SEC", "10"))

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 2


class TavilyClientError(RuntimeError):
    """Raised when Tavily requests fail or are misconfigured."""


async def _backoff_sleep(attempt: int) -> None:
    await asyncio.sleep((0.6 * (2**attempt)) + random.random() * 0.3)


async def search(
    query: str,
    *,
    max_results: int,
    include_answer: bool = False,
    include_raw_content: bool = False,
    search_depth: str = "advanced",
) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        raise TavilyClientError("Missing TAVILY_API_KEY")

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": int(max_results),
        "include_answer": bool(include_answer),
        "include_raw_content": bool(include_raw_content),
        "search_depth": search_depth if search_depth in {"basic", "advanced"} else "advanced",
    }

    timeout = httpx.Timeout(TAVILY_TIMEOUT_SEC)
    last_exc: Exception | None = None

    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(TAVILY_API_URL, json=payload, headers=headers)
                if response.status_code in RETRY_STATUS_CODES:
                    if attempt < MAX_RETRIES:
                        await _backoff_sleep(attempt)
                        continue
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_exc = exc
            except httpx.HTTPError as exc:
                last_exc = exc
                status = getattr(exc.response, "status_code", None)
                if status in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                    await _backoff_sleep(attempt)
                    continue
                break

            if attempt < MAX_RETRIES:
                await _backoff_sleep(attempt)

    raise TavilyClientError(f"Tavily request failed: {last_exc}")
