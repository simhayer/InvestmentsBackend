# services/market_service.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, cast
from datetime import datetime, timezone, timedelta
from services.helpers.linkup.linkup_summary import get_linkup_market_summary
from models.market_summary import MarketSummary
import hashlib
import redis
REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL) if REDIS_URL else None
CACHE_KEY = "linkup:market_summary"
TTL_SEC = 1800  # 30 min

from sqlalchemy.orm import Session

from . import yahoo_service as yq
from services.helpers.db.market_db_service import (
    db_read_latest,
    db_upsert_latest,
    db_append_history,
)

Json = Dict[str, Any]

# ---------------------------
# Config
# ---------------------------
# US-focused top bar
INDEX_META: Dict[str, Tuple[str, str, str | None]] = {
    "^GSPC":  ("SPX",  "S&P 500",   "USD"),
    "^DJI":   ("DJI",  "Dow Jones", "USD"),
    "^IXIC":  ("IXIC", "Nasdaq",    "USD"),
    "BTC-USD":("BTC",  "BTC/USD",   "USD"),
}
SYMBOLS: List[str] = list(INDEX_META.keys())

# Light sparkline config
def _sparkline_params(symbol: str) -> tuple[str, str]:
    s = symbol.upper()
    if s == "BTC-USD":
        # Crypto is 24/7; hourly keeps it lively but still light
        return ("3d", "1h")
    # Indices: daily is tiny payload (≈ 5 points)
    return ("5d", "1d")

# In-memory TTL cache (very small + optional)
_MEM: dict[str, tuple[datetime, Json]] = {}
MEM_KEY = "market:overview:v1"


# ---------------------------
# Helpers
# ---------------------------
def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None

def _as_spark(points: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for p in points or []:
        c = p.get("c")
        if c is not None:
            try:
                out.append(float(c))
            except Exception:
                pass
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

    price = _safe_float(quote.get("current_price"))
    prev  = _safe_float(quote.get("previous_close"))
    change_abs = (price - prev) if (price is not None and prev is not None) else None
    change_pct = _safe_float(quote.get("day_change_pct"))
    currency   = ccy_override or quote.get("currency") or "USD"

    spark = []
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
        hist  = yq.get_price_history(sym, period, interval)
        items.append(_build_item(sym, quote, hist))

    payload: Json = {
        "symbols": SYMBOLS,
        "items": items,
        "ai_summary": None,  # attach later when you generate it
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


# ---------------------------
# Public API (sync; SQLAlchemy Session)
# ---------------------------
def get_market_overview_cached(db: Session, *, max_age_sec: int = 60) -> Json:
    # 1) Memory
    mem_hit = _MEM.get(MEM_KEY)
    if mem_hit:
        ts, payload = mem_hit
        if (datetime.now(timezone.utc) - ts) <= timedelta(seconds=max_age_sec):
            return payload

    # 2) DB
    max_age_sec_for_db = max_age_sec
    db_latest = db_read_latest(db)
    if db_latest:
        fetched_at = db_latest.get("fetched_at")
        is_stale = True
        try:
            if fetched_at:
                ft = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                is_stale = (datetime.now(timezone.utc) - ft) > timedelta(seconds=max_age_sec_for_db)
        except Exception:
            pass

        # If data fresh but AI summary stale/missing, refresh just the summary
        if not is_stale and _needs_ai_refresh(db_latest):
            try:
                db_upsert_latest(db, db_latest)
                _MEM[MEM_KEY] = (datetime.now(timezone.utc), db_latest)
            except Exception:
                # ignore AI failures
                pass
            return db_latest

        if not is_stale:
            _MEM[MEM_KEY] = (datetime.now(timezone.utc), db_latest)
            return db_latest

    # 3) Build fresh, then attach AI summary, persist + history
    fresh = build_overview_payload()

    db_upsert_latest(db, fresh)
    db_append_history(db, fresh)
    _MEM[MEM_KEY] = (datetime.now(timezone.utc), fresh)
    return fresh


# in refresh_market_overview()
def refresh_market_overview(db: Session) -> Json:
    fresh = build_overview_payload()
    db_upsert_latest(db, fresh)
    db_append_history(db, fresh)
    _MEM[MEM_KEY] = (datetime.now(timezone.utc), fresh)
    return fresh

def _needs_ai_refresh(db_latest: Json, *, ttl_minutes: int = 180) -> bool:
    # Refresh if missing or older than ttl
    try:
        meta = db_latest.get("ai_meta") or {}
        ts = meta.get("generated_at")
        if not db_latest.get("ai_summary") or not ts:
            return True
        from datetime import datetime, timezone
        prev = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - prev) > timedelta(minutes=ttl_minutes)
    except Exception:
        return True

def _etag_for(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def get_market_summary_cached(db: Session) -> Tuple[Json, datetime]:
    ...
    latest = (
        db.query(MarketSummary)
        .order_by(MarketSummary.created_at.desc())
        .limit(1)
        .one_or_none()
    )
    if latest and (datetime.now(timezone.utc) - cast(datetime, latest.created_at)) < timedelta(seconds=TTL_SEC):
        return latest.payload, cast(datetime, latest.created_at)

    fresh = get_linkup_market_summary()
    rec = MarketSummary(
        as_of=datetime.fromisoformat(fresh["as_of"]),
        market=fresh["market"],
        payload=fresh,
    )
    db.add(rec)
    db.commit()
    if r:
        r.setex(CACHE_KEY, TTL_SEC, json.dumps(fresh))
    return fresh, cast(datetime, rec.created_at)