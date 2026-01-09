# finnhub_news_sdk.py
from __future__ import annotations

import os
import asyncio
import datetime as dt
from typing import List, Dict, Optional, TypedDict, Any
import finnhub  # pip install finnhub-python


# -------- Public item shape (ready for your UI) --------
class NewsItem(TypedDict, total=False):
    title: str
    url: str
    snippet: Optional[str]
    published_at: Optional[str]  # ISO8601 UTC
    source: Optional[str]
    image: Optional[str]


# -------- Internal helpers --------
def _require_api_key() -> str:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY is missing. Set it in your environment."
        )
    return key


def _unix_to_iso(ts: Any) -> Optional[str]:
    try:
        return dt.datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
    except Exception:
        return None


def _normalize_one(raw: Dict[str, Any]) -> NewsItem:
    # Finnhub company_news schema:
    # {headline, url, summary, datetime, source, image, related, category, ...}
    item: NewsItem = {
        "title": raw.get("headline", "") or "",
        "url": raw.get("url") or "",
        "snippet": raw.get("summary"),
        "published_at": _unix_to_iso(raw.get("datetime")),
        "source": raw.get("source"),
        "image": raw.get("image"),
    }
    return item


# -------- Sync worker (called inside a thread) --------
def _fetch_company_news_blocking(symbol: str, frm: str, to: str, api_key: str) -> List[NewsItem]:
    client = finnhub.Client(api_key=api_key)
    data = client.company_news(symbol, _from=frm, to=to) or []
    items = [_normalize_one(d) for d in data if d.get("url")]
    # newest first
    items.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return items


# -------- Public async API --------
async def get_company_news_for_symbols(
    symbols: List[str],
    *,
    days_back: int = 7,
    limit_per_symbol: int = 6,
    concurrency: int = 6,
) -> Dict[str, List[NewsItem]]:
    """
    Fetch recent company news for multiple symbols using Finnhub's SDK.

    Returns:
        { "AAPL": [NewsItem, ...], "MSFT": [...], ... }
    """
    if not symbols:
        return {}

    api_key = _require_api_key()
    today = dt.date.today()
    frm, to = (today - dt.timedelta(days=days_back)).isoformat(), today.isoformat()

    out: Dict[str, List[NewsItem]] = {}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(sym: str):
        async with sem:
            try:
                items = await asyncio.to_thread(_fetch_company_news_blocking, sym, frm, to, api_key)
            except Exception:
                items = []
            out[sym] = items[:limit_per_symbol] if limit_per_symbol else items

    await asyncio.gather(*[_one(s) for s in symbols])
    return out


# -------- Convenience: single symbol (async) --------
async def get_company_news(
    symbol: str, *, days_back: int = 7, limit: int = 6
) -> List[NewsItem]:
    res = await get_company_news_for_symbols([symbol], days_back=days_back, limit_per_symbol=limit)
    return res.get(symbol, [])
