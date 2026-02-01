# services/finnhub/finnhub_profile_cache.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from services.cache.cache_backend import cache_get_many, cache_set_many
from services.finnhub.finnhub_service import FinnhubService

TTL_PROFILE_SEC = 7 * 24 * 3600  # 7 days


def _ck_profile(symbol: str) -> str:
    return f"FINNHUB:PROFILE:{(symbol or '').strip().upper()}"


async def fetch_profiles_cached(
    finnhub: FinnhubService,
    symbols: List[str],
    *,
    max_concurrency: int = 8,
    ttl_seconds: int = TTL_PROFILE_SEC,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns { "AAPL": {...profile...}, ... } for symbols.
    Uses cache_get_many/cache_set_many + bounded concurrency.
    """
    clean = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    if not clean:
        return {}

    cache_keys = [_ck_profile(s) for s in clean]
    cached = cache_get_many(cache_keys) or {}

    out: Dict[str, Dict[str, Any]] = {}
    misses: List[str] = []

    for sym, ck in zip(clean, cache_keys):
        hit = cached.get(ck)
        if isinstance(hit, dict) and hit:
            out[sym] = hit
        else:
            misses.append(sym)

    if not misses:
        return out

    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def fetch_one(sym: str) -> tuple[str, Dict[str, Any]]:
        async with sem:
            prof = await finnhub.fetch_profile(sym)
            return sym, (prof if isinstance(prof, dict) else {})

    results = await asyncio.gather(*[fetch_one(s) for s in misses], return_exceptions=True)

    write_back: Dict[str, Any] = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        sym, prof = res
        if prof:
            out[sym] = prof
            write_back[_ck_profile(sym)] = prof

    if write_back:
        cache_set_many(write_back, ttl_seconds=ttl_seconds)

    return out
