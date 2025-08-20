# services/yahoo_service.py
from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Callable, List, Optional, Tuple, Type

from yahooquery import Ticker

Number = Optional[float]
Json = Dict[str, Any]

# ---------------------------
# Retry helper (fixed)
# ---------------------------
def retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    delay: float = 0.4,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """
    Retry a function up to `attempts` times with exponential backoff.
    Raises RuntimeError (chained) if all attempts fail.
    """
    attempts = max(1, attempts)
    err: BaseException | None = None

    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:   # <-- exceptions is a tuple of classes
            err = e
            if i < attempts - 1:
                time.sleep(delay * (backoff ** i))
            else:
                break

    raise RuntimeError(f"retry failed after {attempts} attempts") from err


# ---------------------------
# Tiny TTL cache (optional but helpful)
# ---------------------------
_CACHE: Dict[Tuple[str, bool], Tuple[float, Json]] = {}
_CACHE_TTL_SEC = 60  # adjust 30–120s as you like


def _cache_get(symbol: str, include_news: bool) -> Optional[Json]:
    key = (symbol.upper(), include_news)
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts <= _CACHE_TTL_SEC:
        return payload
    _CACHE.pop(key, None)
    return None


def _cache_set(symbol: str, include_news: bool, payload: Json) -> None:
    _CACHE[(symbol.upper(), include_news)] = (time.time(), payload)


# ---------------------------
# Parsing helpers
# ---------------------------
def _ensure_symbol_dict(obj: Any, sym: str) -> Dict[str, Any]:
    """
    yahooquery can return strings, lists, or dicts not keyed by symbol.
    Normalize to a dict (or {}) for the symbol.
    """
    if isinstance(obj, dict):
        # Prefer nested by symbol if present, else accept obj as-is
        if sym in obj and isinstance(obj[sym], dict):
            return obj[sym]
        return obj
    return {}


def _fnum(x: Any) -> Number:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _pct(cur: Number, prev: Number) -> Number:
    try:
        if cur is None or prev in (None, 0):
            return None
        return (cur / prev - 1.0) * 100.0
    except Exception:
        return None


def _dist_pct(cur: Number, ref: Number) -> Number:
    try:
        if cur is None or ref in (None, 0):
            return None
        return (cur - ref) / ref * 100.0
    except Exception:
        return None


def _iso_utc_from_ts(ts: Any) -> Optional[str]:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


# ---------------------------
# Main API
# ---------------------------
def get_full_stock_data(symbol: str, include_news: bool = True) -> Json:
    """
    Fetch quotes + fundamentals (and optionally top news) for `symbol`.
    Returns a stable JSON shape with computed metrics and data quality info.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {
            "status": "error",
            "error_code": "EMPTY_SYMBOL",
            "message": "Symbol is required",
        }

    # Serve from short TTL cache if available
    cached = _cache_get(sym, include_news)
    if cached:
        return cached

    try:
        # NOTE: yahooquery handles session/crumb internally; retries help
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)

        # Fetch raw payloads with retries (handles CSRF/crumb intermittency)
        summary_detail_raw = retry(lambda: tq.summary_detail)
        financial_data_raw = retry(lambda: tq.financial_data)
        key_stats_raw = retry(lambda: tq.key_stats)
        price_raw = retry(lambda: tq.price)

        # Normalize to dicts; never call .get on a non-dict again
        summary_detail = _ensure_symbol_dict(summary_detail_raw, sym)
        financial_data = _ensure_symbol_dict(financial_data_raw, sym)
        key_stats = _ensure_symbol_dict(key_stats_raw, sym)
        price_data = _ensure_symbol_dict(price_raw, sym)

        # --- Core fields
        short_name = price_data.get("shortName") or price_data.get("longName")
        currency = price_data.get("currency")
        exchange = price_data.get("exchangeName") or price_data.get("fullExchangeName")

        current = _fnum(
            summary_detail.get("regularMarketPrice")
            or price_data.get("regularMarketPrice")
        )
        previous = _fnum(
            price_data.get("regularMarketPreviousClose")
            or summary_detail.get("previousClose")
        )

        # 52-week range
        high_52 = _fnum(summary_detail.get("fiftyTwoWeekHigh"))
        low_52 = _fnum(summary_detail.get("fiftyTwoWeekLow"))

        # Fundamentals
        pe_ratio = _fnum(summary_detail.get("trailingPE"))
        forward_pe = _fnum(summary_detail.get("forwardPE") or financial_data.get("forwardPE"))
        price_to_book = _fnum(key_stats.get("priceToBook") or summary_detail.get("priceToBook"))
        beta = _fnum(summary_detail.get("beta") or key_stats.get("beta"))
        dividend_yield = _fnum(summary_detail.get("dividendYield"))
        market_cap = _fnum(summary_detail.get("marketCap") or price_data.get("marketCap"))

        return_on_equity = _fnum(financial_data.get("returnOnEquity"))
        profit_margins = _fnum(financial_data.get("profitMargins"))
        earnings_growth = _fnum(financial_data.get("earningsGrowth"))
        revenue_growth = _fnum(financial_data.get("revenueGrowth"))
        recommendation = _fnum(financial_data.get("recommendationMean"))
        recommendation_key = financial_data.get("recommendationKey")
        target_price = _fnum(financial_data.get("targetMeanPrice"))

        # Quote time
        quote_ts = (
            price_data.get("regularMarketTime")
            or price_data.get("postMarketTime")
            or price_data.get("preMarketTime")
        )
        quote_time_utc = _iso_utc_from_ts(quote_ts)

        # Computed deltas
        day_change = (current - previous) if (current is not None and previous is not None) else None
        day_change_pct = _pct(current, previous)
        distance_from_52w_high_pct = _dist_pct(current, high_52)
        distance_from_52w_low_pct = _dist_pct(current, low_52)

        # Optional news (defensive parsing)
        news_items: List[Dict[str, Any]] = []
        if include_news:
            try:
                raw_news = retry(lambda: tq.news(5) or [])
                for item in raw_news:
                    if not isinstance(item, dict) or "title" not in item:
                        continue
                    ts = item.get("provider_publish_time")
                    thumb = item.get("thumbnail")
                    # Sometimes thumbnail is a dict of resolutions
                    if isinstance(thumb, dict):
                        thumb = (thumb.get("resolutions") or [{}])[0].get("url")
                    news_items.append(
                        {
                            "title": item.get("title"),
                            "summary": item.get("summary"),
                            "url": item.get("url"),
                            "author": item.get("author_name"),
                            "source": item.get("provider_name"),
                            "published_at": _iso_utc_from_ts(ts),
                            "thumbnail": thumb,
                        }
                    )
            except Exception:
                # Non-fatal: keep going without news
                news_items = []

        # Data quality
        missing_fields = [
            k
            for k, v in {
                "current_price": current,
                "currency": currency,
                "previous_close": previous,
            }.items()
            if v is None
        ]
        is_stale = False
        if quote_time_utc:
            try:
                qt = datetime.fromisoformat(quote_time_utc.replace("Z", "+00:00"))
                is_stale = (datetime.now(timezone.utc) - qt) > timedelta(days=2)
            except Exception:
                pass

        payload: Json = {
            "status": "ok",
            "symbol": sym,
            "name": short_name,
            "currency": currency,
            "exchange": exchange,
            "quote_time_utc": quote_time_utc,
            # Prices
            "current_price": current,
            "previous_close": previous,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            # Range
            "52_week_high": high_52,
            "52_week_low": low_52,
            "distance_from_52w_high_pct": distance_from_52w_high_pct,
            "distance_from_52w_low_pct": distance_from_52w_low_pct,
            # Fundamentals
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "forward_pe": forward_pe,
            "price_to_book": price_to_book,
            "beta": beta,
            "dividend_yield": dividend_yield,
            "return_on_equity": return_on_equity,
            "profit_margins": profit_margins,
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth,
            "recommendation": recommendation,
            "recommendation_key": recommendation_key,
            "target_price": target_price,
            # News
            "news": news_items,
            # Quality & provenance
            "data_quality": {
                "source": "Yahoo Finance via yahooquery",
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "is_stale": is_stale,
                "missing_fields": missing_fields,
            },
        }

        _cache_set(sym, include_news, payload)
        return payload

    except Exception as e:
        # Hardened error surface (covers CSRF/crumb & shape mismatches)
        return {
            "status": "error",
            "error_code": "YAHOOQUERY_FAILURE",
            "message": f"Failed to fetch data for {sym}: {str(e)}",
        }
