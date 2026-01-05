from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Iterable, List, Optional, TypedDict
from urllib.parse import urlparse

from dateutil import parser

from services.tavily.client import TavilyClientError, search as tavily_search

FILINGS_MAX_ITEMS = int(
    os.getenv("FILINGS_MAX_ITEMS", os.getenv("SINGLE_STOCK_FILINGS_MAX_ITEMS", "6"))
)
FILINGS_SNIPPET_MAX_CHARS = int(os.getenv("FILINGS_SNIPPET_MAX_CHARS", "500"))

FILINGS_KEYWORDS = (
    "filing",
    "8-k",
    "10-q",
    "10-k",
    "sec",
    "guidance",
    "risk factors",
    "material event",
)


class FilingItem(TypedDict):
    id: str
    title: str
    url: str
    source: str
    published_at: Optional[str]
    snippet: str


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return "unknown"
    return domain[4:] if domain.startswith("www.") else domain or "unknown"


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parser.parse(value)
    except Exception:
        return None


def needs_filings_for_request(user_request: Optional[str], explicit: bool) -> bool:
    if explicit:
        return True
    if not user_request:
        return False
    lowered = user_request.lower()
    return any(keyword in lowered for keyword in FILINGS_KEYWORDS)


def normalize_filing_results(
    results: Iterable[dict[str, Any]],
    *,
    max_items: int = FILINGS_MAX_ITEMS,
) -> List[FilingItem]:
    parsed: list[dict[str, Any]] = []
    for order, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        title = _normalize_text(item.get("title") or item.get("headline") or "")
        url = _normalize_text(item.get("url") or item.get("link") or "")
        if not title or not url:
            continue
        snippet = _normalize_text(
            item.get("content")
            or item.get("snippet")
            or item.get("description")
            or ""
        )
        if not snippet:
            snippet = title
        published_at = _normalize_text(
            item.get("published_date")
            or item.get("published")
            or item.get("date")
            or ""
        )
        parsed.append(
            {
                "title": title,
                "url": url,
                "source": _extract_domain(url),
                "published_at": published_at or None,
                "_published_dt": _parse_datetime(published_at),
                "_order": order,
                "snippet": _truncate(snippet, FILINGS_SNIPPET_MAX_CHARS),
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed:
        key = item.get("url", "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(
        key=lambda row: (
            0 if row.get("_published_dt") else 1,
            -row["_published_dt"].timestamp() if row.get("_published_dt") else row["_order"],
        )
    )

    out: list[FilingItem] = []
    for idx, item in enumerate(deduped[: max(0, int(max_items))], start=1):
        out.append(
            {
                "id": f"filing_{idx}",
                "title": item["title"],
                "url": item["url"],
                "source": item["source"],
                "published_at": item["published_at"],
                "snippet": item["snippet"],
            }
        )
    return out


async def fetch_filings(
    symbol: str,
    *,
    needs_filings: bool,
    cik: Optional[str] = None,
) -> List[FilingItem]:
    if not needs_filings:
        return []

    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol and not cik:
        return []

    queries = [
        f"{clean_symbol} site:sec.gov 8-K",
        f"{clean_symbol} site:sec.gov 10-Q",
        f"{clean_symbol} site:sec.gov 10-K",
    ]
    if cik:
        queries.append(f"{cik} site:sec.gov 8-K")

    tasks = [
        tavily_search(
            query=q,
            max_results=FILINGS_MAX_ITEMS,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
        )
        for q in queries
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict[str, Any]] = []
    for resp in responses:
        if isinstance(resp, Exception):
            continue
        if isinstance(resp, dict):
            results = resp.get("results")
            if isinstance(results, list):
                combined.extend(results)

    try:
        return normalize_filing_results(combined, max_items=FILINGS_MAX_ITEMS)
    except TavilyClientError:
        return []
