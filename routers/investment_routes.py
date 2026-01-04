# routers/market_routes.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, Optional

from services.yahoo_service import (
    get_full_stock_data,
    get_price_history,
    get_financials,
    get_earnings,
    get_analyst,
    get_overview,
)

from services.cache.cache_backend import cache_get, cache_set
router = APIRouter()
Json = Dict[str, Any]

TTL_YAHOO_QUOTE_SEC = 60
TTL_YAHOO_HISTORY_SEC = 15 * 60           # 15m
TTL_YAHOO_EARNINGS_SEC = 12 * 60 * 60     # 12h
TTL_YAHOO_FINANCIALS_SEC = 48 * 60 * 60   # 48h
TTL_YAHOO_PROFILE_SEC = 24 * 60 * 60      # 24h (overview + analyst)

def _norm_sym(s: str | None) -> str:
    return (s or "").upper().strip()

def _ck(kind: str, *parts: str) -> str:
    clean = [p.strip() for p in parts if p and p.strip()]
    return "yahoo:" + kind + ":" + ":".join(clean)

def _ok_or_raise(data: Any, *, default_msg: str = "Fetch failed") -> Json:
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON")
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", default_msg))
    return data  # type: ignore[return-value]

@router.get("/quote/{symbol}")
async def get_quote(symbol: str, q: Optional[str] = Query(default=None)):
    yahoo_symbol = _norm_sym(q or symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("quote", yahoo_symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_full_stock_data, yahoo_symbol)
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON")

    if data.get("status") != "ok":
        msg = data.get("message") or "Failed to fetch quote data"
        code = 502 if data.get("error_code") == "YAHOOQUERY_FAILURE" else 400
        raise HTTPException(status_code=code, detail=msg)

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_QUOTE_SEC)
    return data


@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    q: Optional[str] = Query(default=None),
    period: str = Query("1y", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"),
):
    yahoo_symbol = _norm_sym(q or symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("history", yahoo_symbol, period, interval)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_price_history, yahoo_symbol, period, interval)
    data = _ok_or_raise(data, default_msg="Fetch failed")

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_HISTORY_SEC)
    return data


@router.get("/financials/{symbol}")
async def financials(
    symbol: str,
    period: str = Query("annual", pattern="^(annual|quarterly)$"),
):
    yahoo_symbol = _norm_sym(symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("financials", yahoo_symbol, period)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_financials, yahoo_symbol, period)
    data = _ok_or_raise(data, default_msg="Fetch failed")

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_FINANCIALS_SEC)
    return data


@router.get("/earnings/{symbol}")
async def earnings(symbol: str):
    yahoo_symbol = _norm_sym(symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("earnings", yahoo_symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_earnings, yahoo_symbol)
    data = _ok_or_raise(data, default_msg="Fetch failed")

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_EARNINGS_SEC)
    return data


@router.get("/analyst/{symbol}")
async def analyst(symbol: str):
    yahoo_symbol = _norm_sym(symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("analyst", yahoo_symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_analyst, yahoo_symbol)
    data = _ok_or_raise(data, default_msg="Fetch failed")

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_PROFILE_SEC)
    return data


@router.get("/overview/{symbol}")
async def overview(symbol: str):
    yahoo_symbol = _norm_sym(symbol)
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = _ck("overview", yahoo_symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    data = await run_in_threadpool(get_overview, yahoo_symbol)
    data = _ok_or_raise(data, default_msg="Fetch failed")

    cache_set(cache_key, data, ttl_seconds=TTL_YAHOO_PROFILE_SEC)
    return data
