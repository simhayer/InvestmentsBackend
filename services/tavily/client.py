from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Dict, List
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
    search_depth: str = "advanced"
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
        "topic": "finance",
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
                status = None
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                if status in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                    await _backoff_sleep(attempt)
                    continue
                break

            if attempt < MAX_RETRIES:
                await _backoff_sleep(attempt)

    raise TavilyClientError(f"Tavily request failed: {last_exc}")

# ----------------------------
# 6) Tavily compaction (smaller context for LLM)
# ----------------------------
def compact_results(results: Any, limit: int = 6) -> str:
    """
    Tavily returns a dict-like structure. We keep only the top items (title/url/snippet).
    If it's already a string, return as-is (but truncated).
    """
    if results is None:
        return ""

    if isinstance(results, str):
        return results[:8000]

    if isinstance(results, dict):
        items = results.get("results") or results.get("data") or []
        if not isinstance(items, list):
            return str(results)[:8000]

        out_lines: List[str] = []
        for r in items[: max(1, limit)]:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or r.get("snippet") or "").strip()
            if content:
                content = content.replace("\n", " ").strip()
            line = f"- {title}\n  {url}\n  {content}"
            out_lines.append(line)

        return "\n".join(out_lines)[:8000]

    return str(results)[:8000]