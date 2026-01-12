from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx
from services.finnhub.finnhub_service import FinnhubService, FinnhubServiceError
from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.client import FINNHUB_CLIENT
from utils.common_helpers import timed
import logging
logger = logging.getLogger(__name__)

TTL_FUNDAMENTALS_SEC = 600
TTL_METRICS_SEC = 24*3600

def _ck_fund(symbol: str) -> str:
    return f"FUNDAMENTALS:{(symbol or '').strip().upper()}"

@dataclass(frozen=True)
class FundamentalsResult:
    data: Dict[str, Any]
    gaps: List[str]

def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _pick_metric(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics:
            val = _coerce_float(metrics.get(key))
            if val is not None:
                return val
    return None

async def fetch_fundamentals(symbol: str, *, timeout_s: float = 5.0) -> FundamentalsResult:
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        return FundamentalsResult({}, ["Missing symbol for fundamentals"])

    gaps: list[str] = []

    try:
        svc = FinnhubService(timeout=timeout_s)
    except FinnhubServiceError as exc:
        return FundamentalsResult({}, [str(exc)])
    
    client = FINNHUB_CLIENT

    results = await asyncio.gather(
        svc.fetch_profile(clean_symbol, client=client),
        svc.fetch_quote(clean_symbol, client=client),
        svc.fetch_basic_financials(clean_symbol, client=client),
        svc.fetch_earnings(clean_symbol, client=client),
        return_exceptions=True,
    )

    with timed("fh_profile", logger, state={"symbol": clean_symbol, "task_id": "fundamentals"}):
        profile = await svc.fetch_profile(clean_symbol, client=client)

    with timed("fh_quote", logger, state={"symbol": clean_symbol, "task_id": "fundamentals"}):
        quote = await svc.fetch_quote(clean_symbol, client=client)

    with timed("fh_metrics", logger, state={"symbol": clean_symbol, "task_id": "fundamentals"}):
        metrics_raw = await svc.fetch_basic_financials(clean_symbol, client=client)

    with timed("fh_earnings", logger, state={"symbol": clean_symbol, "task_id": "fundamentals"}):
        earnings_raw = await svc.fetch_earnings(clean_symbol, client=client)

    profile, quote, metrics_raw, earnings_raw = results

    if isinstance(profile, Exception) or not isinstance(profile, dict) or not profile:
        gaps.append("Company profile unavailable")
        profile = {}
    if isinstance(quote, Exception) or not isinstance(quote, dict) or not quote:
        gaps.append("Price snapshot unavailable")
        quote = {}

    metrics: Dict[str, Any] = {}
    if isinstance(metrics_raw, Exception) or not isinstance(metrics_raw, dict) or not metrics_raw:
        gaps.append("Key metrics unavailable")
    else:
        metrics = metrics_raw.get("metric") or {} if isinstance(metrics_raw.get("metric"), dict) else metrics_raw or {}

    earnings: List[Dict[str, Any]] = []
    if isinstance(earnings_raw, Exception) or not isinstance(earnings_raw, list):
        earnings = []
    else:
        earnings = [row for row in earnings_raw if isinstance(row, dict)]

    normalized = {
        "market_cap": _pick_metric(metrics, "marketCapitalization")
        or _coerce_float(profile.get("marketCapitalization")),
        "pe_ttm": _pick_metric(metrics, "peTTM"),
        "revenue_growth_yoy": _pick_metric(metrics, "revenueGrowthTTM", "revenueGrowthTTMYoy"),
        "gross_margin": _pick_metric(metrics, "grossMarginTTM"),
        "operating_margin": _pick_metric(metrics, "operatingMarginTTM"),
        "free_cash_flow": _pick_metric(metrics, "freeCashFlowTTM"),
        "debt_to_equity": _pick_metric(metrics, "totalDebtToEquity", "debtToEquity"),
    }

    data = {
        "symbol": clean_symbol,
        "profile": profile,
        "quote": quote,
        "metrics": metrics,
        "earnings": earnings,
        "normalized": normalized,
    }

    return FundamentalsResult(data=data, gaps=gaps)



async def fetch_fundamentals_cached(symbol: str, *, timeout_s: float = 5.0, ttl_seconds: int = TTL_FUNDAMENTALS_SEC) -> FundamentalsResult:
    """
    Optional cached wrapper you can call from anywhere.
    """
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        return FundamentalsResult({}, ["Missing symbol for fundamentals"])

    key = _ck_fund(clean_symbol)
    cached = cache_get(key)
    if isinstance(cached, dict) and "data" in cached and "gaps" in cached:
        data = cached.get("data") or {}
        gaps = cached.get("gaps") or []
        if isinstance(data, dict) and isinstance(gaps, list):
            return FundamentalsResult(data=data, gaps=gaps)

    res = await fetch_fundamentals(clean_symbol, timeout_s=timeout_s)
    cache_set(key, {"data": res.data, "gaps": res.gaps}, ttl_seconds=ttl_seconds)
    return res