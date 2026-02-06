# services/market_service.py
from __future__ import annotations

import json
import os
import math
import hashlib
from typing import Any, Dict, List, Tuple, cast
from datetime import datetime, timezone, timedelta
import redis
from sqlalchemy.orm import Session
from models.market_summary import MarketSummary
from . import yahoo_service as yq
from services.helpers.market_db_service import (
    db_read_latest,
    db_upsert_latest,
    db_append_history,
)
from utils.common_helpers import safe_float

REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL) if REDIS_URL else None
CACHE_KEY = "linkup:market_summary"
TTL_SEC = 1800  # 30 min

Json = Dict[str, Any]

# ---------------------------
# Config
# ---------------------------
# US-focused top bar
INDEX_META: Dict[str, Tuple[str, str, str | None]] = {
    "^GSPC": ("SPX", "S&P 500", "USD"),
    "^DJI": ("DJI", "Dow Jones", "USD"),
    "^IXIC": ("IXIC", "Nasdaq", "USD"),
    "BTC-USD": ("BTC", "BTC/USD", "USD"),
}
SYMBOLS: List[str] = list(INDEX_META.keys())

# In-memory TTL cache (very small + optional)
_MEM: dict[str, tuple[datetime, Json]] = {}
MEM_KEY = "market:overview:v1"


# ---------------------------
# Sanitization helpers
# ---------------------------
def _is_nonfinite_number(x: Any) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def sanitize_json(obj: Any) -> Any:
    """
    Recursively replace NaN/±Inf -> None, convert numpy/pandas scalars,
    and make arrays JSON-safe. This ensures Postgres JSONB accepts the payload.
    """
    # Optional: numpy support without hard dependency
    # try:
    #     import numpy as np  # type: ignore
    #     if isinstance(obj, (np.floating, np.integer)):
    #         obj = float(obj)
    #     if isinstance(obj, np.ndarray):
    #         obj = obj.tolist()
    # except Exception:
    #     pass

    # Optional: pandas NA/NaT handling without hard dependency
    try:
        import pandas as pd  # type: ignore
        if obj is pd.NaT:  # noqa: E721
            return None
        try:
            # pd.isna works for many scalar types
            if pd.isna(obj):  # type: ignore[attr-defined]
                return None
        except Exception:
            pass
    except Exception:
        pass

    if isinstance(obj, float):
        return None if _is_nonfinite_number(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(v) for v in obj]
    return obj


# ---------------------------
# Data shaping helpers
# ---------------------------

def _sparkline_params(symbol: str) -> tuple[str, str]:
    s = symbol.upper()
    if s == "BTC-USD":
        # Crypto is 24/7; hourly keeps it lively but still light
        return ("3d", "1h")
    # Indices: daily is tiny payload
    return ("5d", "1d")


def _as_spark(points: List[Dict[str, Any]]) -> List[float]:
    """
    Build a compact sparkline with ONLY finite floats.
    Drop NaN/Inf rather than emitting nulls for a tiny array.
    """
    out: List[float] = []
    for p in points or []:
        c = p.get("c")
        if c is None:
            continue
        try:
            v = float(c)
            if not math.isfinite(v):
                continue
            out.append(v)
        except Exception:
            continue
    # hard cap to avoid huge payloads
    return out[-60:]


def _build_item(symbol: str, quote: Json, hist: Json) -> Json:
    key, label, ccy_override = INDEX_META.get(symbol, (symbol, symbol, None))

    if quote.get("status") != "ok":
        return {
            "key": key,
            "label": label,
            "symbol": symbol,
            "price": None,
            "changeAbs": None,
            "changePct": None,
            "currency": ccy_override or "USD",
            "sparkline": [],
            "error": quote.get("message") or quote.get("error_code"),
        }

    price = safe_float(quote.get("current_price"))
    prev = safe_float(quote.get("previous_close"))
    change_abs = (price - prev) if (price is not None and prev is not None) else None
    change_pct = safe_float(quote.get("day_change_pct"))
    currency = ccy_override or quote.get("currency") or "USD"

    spark: List[float] = []
    if hist.get("status") == "ok":
        spark = _as_spark(hist.get("points", []))
        # Fallback if Yahoo returned too sparse a set
        if len(spark) < 2:
            alt_period, alt_interval = ("1mo", "1d") if symbol != "BTC-USD" else ("3d", "90m")
            hist2 = yq.get_price_history(symbol, alt_period, alt_interval)
            if hist2.get("status") == "ok":
                spark2 = _as_spark(hist2.get("points", []))
                spark = spark if len(spark) >= len(spark2) else spark2

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


# ---------------------------
# Core builders
# ---------------------------
def build_overview_payload() -> Json:
    items: List[Json] = []
    for sym in SYMBOLS:
        quote = yq.get_full_stock_data(sym)
        period, interval = _sparkline_params(sym)
        hist = yq.get_price_history(sym, period, interval)
        items.append(_build_item(sym, quote, hist))

    payload: Json = {
        "symbols": SYMBOLS,
        "items": items,
        "ai_summary": None,  # attach later when you generate it
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


# ---------------------------
# AI refresh meta helper
# ---------------------------
def _needs_ai_refresh(db_latest: Json, *, ttl_minutes: int = 180) -> bool:
    # Refresh if missing or older than ttl
    try:
        meta = db_latest.get("ai_meta") or {}
        ts = meta.get("generated_at")
        if not db_latest.get("ai_summary") or not ts:
            return True
        prev = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - prev) > timedelta(minutes=ttl_minutes)
    except Exception:
        return True


def etag_for(obj: dict) -> str:
    # Only used where you need a deterministic hash of JSON-serializable dicts
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------
# Public API (sync; SQLAlchemy Session)
# ---------------------------
def get_market_overview_cached(db: Session, *, max_age_sec: int = 60) -> Json:
    # 1) Memory cache
    mem_hit = _MEM.get(MEM_KEY)
    if mem_hit:
        ts, payload = mem_hit
        if (datetime.now(timezone.utc) - ts) <= timedelta(seconds=max_age_sec):
            return payload

    # 2) DB cache
    max_age_sec_for_db = max_age_sec
    db_latest = db_read_latest(db)
    if db_latest:
        fetched_at = db_latest.get("fetched_at")
        is_stale = True
        try:
            if fetched_at:
                ft = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
                is_stale = (datetime.now(timezone.utc) - ft) > timedelta(seconds=max_age_sec_for_db)
        except Exception:
            pass

        # If data fresh but AI summary stale/missing, refresh just the summary
        if not is_stale and _needs_ai_refresh(db_latest):
            try:
                # Sanitize before write-back (defensive)
                clean_db_latest = sanitize_json(db_latest)
                json.dumps(clean_db_latest, allow_nan=False)  # fail-fast if anything bad slipped in
                db_upsert_latest(db, clean_db_latest)
                _MEM[MEM_KEY] = (datetime.now(timezone.utc), clean_db_latest)
            except Exception:
                # ignore AI failures
                pass
            return db_latest

        if not is_stale:
            _MEM[MEM_KEY] = (datetime.now(timezone.utc), db_latest)
            return db_latest

    # 3) Build fresh, sanitize, persist + history
    fresh = build_overview_payload()
    clean = sanitize_json(fresh)
    # Guardrail: raise in Python if any NaN/Inf still present
    json.dumps(clean, allow_nan=False)

    db_upsert_latest(db, clean)     # pass Python dict/list to JSONB columns
    db_append_history(db, clean)
    _MEM[MEM_KEY] = (datetime.now(timezone.utc), clean)
    return clean


def refresh_market_overview(db: Session) -> Json:
    fresh = build_overview_payload()
    clean = sanitize_json(fresh)
    json.dumps(clean, allow_nan=False)

    db_upsert_latest(db, clean)
    db_append_history(db, clean)
    _MEM[MEM_KEY] = (datetime.now(timezone.utc), clean)
    return clean