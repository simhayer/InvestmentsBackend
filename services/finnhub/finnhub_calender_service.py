# services/finnhub/finnhub_earnings_calendar_service.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.finnhub_service import FinnhubService  # adjust import path


TTL_EARNINGS_CAL_SEC = int(os.getenv("TTL_EARNINGS_CAL_SEC", "21600"))  # 6 hours
DEFAULT_WINDOW_DAYS = int(os.getenv("EARNINGS_CAL_WINDOW_DAYS", "120"))
DEFAULT_LIMIT = int(os.getenv("EARNINGS_CAL_LIMIT", "6"))


def _utc_today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _utc_plus_days_iso(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=int(days))).isoformat()


def _ck_earnings(symbol: str, from_date: str, to_date: str, international: bool) -> str:
    sym = (symbol or "").strip().upper()
    intl = "1" if international else "0"
    return f"ANALYZE:EARNINGS_CAL:{sym}:{from_date}:{to_date}:I{intl}"


def compact_earnings_calendar(
    raw: Dict[str, Any],
    *,
    symbol: str,
    limit: int = DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Finnhub response:
      {"earningsCalendar": [ {date, epsActual, epsEstimate, revenueActual, revenueEstimate, hour, quarter, year, symbol} ]}

    We keep near-term items and strip noise.
    """
    items = raw.get("earningsCalendar") or []
    if not isinstance(items, list):
        return []

    sym = (symbol or "").strip().upper()

    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if (it.get("symbol") or "").strip().upper() != sym:
            continue

        # keep only fields we need
        out.append({
            "date": it.get("date"),
            "hour": it.get("hour"),  # bmo/amc/dmh
            "quarter": it.get("quarter"),
            "year": it.get("year"),
            "eps_actual": it.get("epsActual"),
            "eps_estimate": it.get("epsEstimate"),
            "revenue_actual": it.get("revenueActual"),
            "revenue_estimate": it.get("revenueEstimate"),
            "symbol": it.get("symbol"),
        })

    # Keep the soonest upcoming first if Finnhub returns in date order.
    # If dates are missing, just return as-is.
    out = [x for x in out if x.get("date")]
    return out[: max(1, int(limit))]


async def get_earnings_calendar_compact_cached(
    *,
    symbol: str,
    from_date: Optional[str] = None,  # YYYY-MM-DD
    to_date: Optional[str] = None,    # YYYY-MM-DD
    window_days: int = DEFAULT_WINDOW_DAYS,
    limit: int = DEFAULT_LIMIT,
    international: bool = False,
    svc: Optional[FinnhubService] = None,
) -> List[Dict[str, Any]]:
    """
    Cached, compact earnings calendar for one symbol.
    Returns [] on failure.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    fd = (from_date or _utc_today_iso()).strip()
    td = (to_date or _utc_plus_days_iso(window_days)).strip()

    key = _ck_earnings(sym, fd, td, international)
    cached = cache_get(key)

    if isinstance(cached, list):
        return cached
    if isinstance(cached, dict) and isinstance(cached.get("items"), list):
        return cached["items"]

    try:
        service = svc or FinnhubService()
        raw = await service.fetch_earnings_calendar(
            symbol=sym,
            from_date=fd,
            to_date=td,
            international=international,
        )
        compact = compact_earnings_calendar(raw, symbol=sym, limit=limit)
        cache_set(key, {"items": compact}, ttl_seconds=TTL_EARNINGS_CAL_SEC)
        return compact
    except Exception:
        return []
