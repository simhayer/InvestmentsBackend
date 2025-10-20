# services/market_service.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from . import yahoo_service as yq

Json = Dict[str, Any]
INDEX_META: dict[str, tuple[str, str, str | None]] = {
    "^GSPC":  ("SPX",  "S&P 500", "USD"),
    "^DJI":   ("DJI",  "Dow Jones", "USD"),
    "^IXIC":  ("IXIC", "Nasdaq", "USD"),
    "BTC-USD":("BTC",  "Bitcoin (USD)", "USD"),
}

# Choose a compact series for sparklines
def _sparkline_params(symbol: str) -> Tuple[str, str]:
    if symbol.upper() == "BTC-USD":
        # Crypto: 1d / 30m gives a nice intraday curve
        return ("1d", "30m")
    # Indices: 1d / 2h is a good balance
    return ("1d", "1h")

def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None

def _as_numbers(points: List[Dict[str, Any]]) -> List[float]:
    # Extract close values, filter None
    out: List[float] = []
    for p in points:
        c = p.get("c")
        if c is not None:
            try:
                out.append(float(c))
            except Exception:
                pass
    return out

async def _fetch_quote(symbol: str) -> Json:
    # yahoo_service uses sync calls; run in thread for concurrency
    return await asyncio.to_thread(yq.get_full_stock_data, symbol)

async def _fetch_history(symbol: str) -> Json:
    period, interval = _sparkline_params(symbol)
    return await asyncio.to_thread(yq.get_price_history, symbol, period, interval)

def _build_item(symbol: str, quote: Json, hist: Json) -> Json:
    meta = INDEX_META.get(symbol, (symbol, symbol, None))
    key, label, ccy_override = meta

    status = quote.get("status")
    if status != "ok":
        # degraded but stable payload
        return {
            "key": key,
            "label": label,
            "symbol": symbol,
            "price": None,
            "changeAbs": None,
            "changePct": None,
            "currency": ccy_override or "USD",
            "sparkline": [],
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "error": quote.get("message") or quote.get("error_code"),
        }

    price = _safe_float(quote.get("current_price"))
    prev = _safe_float(quote.get("previous_close"))
    change_abs = price - prev if (price is not None and prev is not None) else None
    change_pct = _safe_float(quote.get("day_change_pct"))
    currency = (ccy_override or quote.get("currency") or "USD")

    spark = []
    if hist.get("status") == "ok":
        spark = _as_numbers(hist.get("points", []))
        # Optional: keep only last ~60 points to avoid overdraw
        if len(spark) > 60:
            spark = spark[-60:]

    return {
        "key": key,
        "label": label,
        "symbol": symbol,
        "price": price,
        "changeAbs": change_abs,
        "changePct": change_pct,
        "currency": currency,
        "sparkline": spark,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
    }

async def get_market_overview_items() -> List[Json]:
    symbols = ["^GSPC", "^IXIC", "^GSPTSE", "BTC-USD"]

    # Fetch quotes & history concurrently
    quote_tasks = [asyncio.create_task(_fetch_quote(s)) for s in symbols]
    hist_tasks  = [asyncio.create_task(_fetch_history(s)) for s in symbols]

    quotes = await asyncio.gather(*quote_tasks, return_exceptions=True)
    hists  = await asyncio.gather(*hist_tasks,  return_exceptions=True)

    items: List[Json] = []
    for i, sym in enumerate(symbols):
        q = quotes[i]
        h = hists[i]

        # Normalize exceptions into error payloads
        qj: Json = q if isinstance(q, dict) else {"status": "error", "message": str(q)}
        hj: Json = h if isinstance(h, dict) else {"status": "error", "message": str(h)}

        items.append(_build_item(sym, qj, hj))

    return items
