# finnhub_news_sdk.py
from __future__ import annotations

import logging
import os
import asyncio

logger = logging.getLogger(__name__)
import datetime as dt
from typing import List, Dict, Optional, TypedDict, Any
try:
    import finnhub  # pip install finnhub-python
except Exception:
    finnhub = None
from services.cache.cache_backend import cache_get, cache_set

TTL_NEWS_SEC = int(os.getenv("TTL_NEWS_SEC", "900"))  # 15m
TTL_EMPTY_NEWS_SEC = 60

def _ck_news(symbol: str, days_back: int, limit: int) -> str:
    s = (symbol or "").strip().upper()
    return f"ANALYZE:NEWS:{s}:D{days_back}:L{limit}"


def _ck_global_news(category: str, limit: int) -> str:
    return f"ANALYZE:NEWS:GLOBAL:{category}:L{limit}"
    

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
    if finnhub is None:
        raise RuntimeError("finnhub package is not installed")
    client = finnhub.Client(api_key=api_key)
    data = client.company_news(symbol, _from=frm, to=to) or []
    items = [_normalize_one(d) for d in data if d.get("url")]
    # newest first
    items.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return items


def _fetch_general_news_blocking(category: str, api_key: str) -> List[NewsItem]:
    """Fetch general/market news from Finnhub (general_news API)."""
    if finnhub is None:
        raise RuntimeError("finnhub package is not installed")
    client = finnhub.Client(api_key=api_key)
    # general_news(category, min_id=0) - category: general, crypto, forex, merger, etc.
    data = client.general_news(category=category or "general", min_id=0) or []
    items = [_normalize_one(d) for d in data if d.get("url")]
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
    today = dt.datetime.now(dt.timezone.utc).date()
    frm = (today - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')
    to = today.strftime('%Y-%m-%d')

    out: Dict[str, List[NewsItem]] = {}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(sym: str):
        async with sem:
            try:
                items = await asyncio.to_thread(_fetch_company_news_blocking, sym, frm, to, api_key)
            except Exception as e:
                logger.exception("Finnhub company_news failed for %s (%s -> %s)", sym, frm, to)
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


def compact_finnhub_news(items: List[NewsItem], max_items: int = 8) -> str:
    lines = []
    for it in (items or [])[:max_items]:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        if not title or not url:
            continue

        src = (it.get("source") or "").strip()
        dt = (it.get("published_at") or "").strip()
        snip = (it.get("snippet") or "").strip()

        if len(snip) > 220:
            snip = snip[:220] + "…"

        lines.append(f"- [{dt}] {src} | {title} — {snip} ({url})")
    return "\n".join(lines)

async def get_company_news_cached(symbol: str, *, days_back: int = 7, limit: int = 6):
    ck = _ck_news(symbol, days_back, limit)

    cached = cache_get(ck)
    if isinstance(cached, dict) and isinstance(cached.get("items"), list) and isinstance(cached.get("compact"), str):
        return cached

    items = await get_company_news(symbol, days_back=days_back, limit=limit)
    compact = compact_finnhub_news(items, max_items=limit)

    payload = {"items": items, "compact": compact}

    # ✅ don’t “poison” cache with empty payload for 15 min
    ttl = TTL_NEWS_SEC if items else TTL_EMPTY_NEWS_SEC
    cache_set(ck, payload, ttl_seconds=ttl)

    return payload


# -------- Global / general market news (for Finance World page) --------
TTL_GLOBAL_NEWS_SEC = int(os.getenv("TTL_GLOBAL_NEWS_SEC", "600"))  # 10 min

async def get_global_news(
    category: str = "general",
    limit: int = 20,
) -> List[NewsItem]:
    """Fetch general market news (not tied to a symbol). Categories: general, crypto, forex, merger, etc."""
    api_key = _require_api_key()

    def _fetch():
        return _fetch_general_news_blocking(category, api_key)

    items = await asyncio.to_thread(_fetch)
    return (items or [])[:limit]


async def get_global_news_cached(
    category: str = "general",
    limit: int = 20,
    force_refresh: bool = False,
) -> List[NewsItem]:
    """Cached global news for public Finance World page."""
    ck = _ck_global_news(category, limit)
    if not force_refresh:
        cached = cache_get(ck)
        if isinstance(cached, list):
            return cached

    items = await get_global_news(category=category, limit=limit)
    cache_set(ck, items, ttl_seconds=TTL_GLOBAL_NEWS_SEC)
    return items
