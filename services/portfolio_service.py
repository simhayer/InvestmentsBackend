# services/portfolio_service.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from decimal import Decimal
from math import fsum
import time

from sqlalchemy.orm import Session
from services.finnhub_service import FinnhubService
from services.holding_service import get_holdings_with_live_prices_top
from services.plaid_service import get_connections
from models.holding import HoldingOut

Number = float | int | Decimal

def _to_float(x: Number | None) -> float:
    if x is None:
        return 0.0
    if isinstance(x, Decimal):
        return float(x)
    return float(x)

def _position_value(h: HoldingOut) -> float:
    # prefer precomputed value if present and > 0; else compute curr*qty
    v = _to_float(getattr(h, "value", None))
    if v > 0:
        return v
    return _to_float(h.current_price) * _to_float(h.quantity)

def _normalize_alloc(d: Dict[str, float], total: float) -> List[Dict[str, Any]]:
    # sort descending by value
    items = sorted(d.items(), key=lambda kv: -kv[1])
    if total <= 0:
        return [{"key": k, "value": round(v, 2), "weight": None} for k, v in items]
    return [
        {"key": k, "value": round(v, 2), "weight": round(v / total * 100.0, 2)}
        for k, v in items
    ]

async def get_portfolio_summary(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    top_n: int = 5,
) -> Dict[str, Any]:
    # 1) Enrich once (quotes, currency, values)
    enriched = await get_holdings_with_live_prices_top(user_id, db, finnhub, currency=currency, top_n=top_n)
    items: List[HoldingOut] = enriched.get("items", [])

    # Route-level price status (live/mixed/unavailable)
    live_count = sum(1 for it in items if getattr(it, "price_status", None) == "live")
    if live_count == 0:
        price_status = "unavailable"
    elif live_count == len(items):
        price_status = "live"
    else:
        price_status = "mixed"

    # 2) Portfolio totals (fsum for numeric stability)
    qtys   = [_to_float(h.quantity) for h in items]
    currs  = [_to_float(h.current_price) for h in items]
    purs   = [_to_float(h.purchase_price) for h in items]
    pcs    = [_to_float(h.previous_close) if h.previous_close is not None else 0.0 for h in items]

    market_value = fsum(q * c for q, c in zip(qtys, currs))
    cost_basis   = fsum((q * p) for q, p in zip(qtys, purs) if p > 0)
    # day P/L only when previous_close > 0
    day_pl_terms = [(c - pc) * q for q, c, pc in zip(qtys, currs, pcs) if pc > 0]
    day_pl = fsum(day_pl_terms)
    prev_close_total = fsum((pc * q) for q, pc in zip(qtys, pcs) if pc > 0)

    unrealized_pl     = (market_value - cost_basis) if cost_basis > 0 else 0.0
    unrealized_pl_pct = (unrealized_pl / cost_basis * 100.0) if cost_basis > 0 else None
    day_pl_pct        = (day_pl / prev_close_total * 100.0) if prev_close_total > 0 else None

    # 3) Allocations
    alloc_by_type: Dict[str, float] = {}
    alloc_by_account: Dict[str, float] = {}

    for h in items:
        val = _position_value(h)
        t = (h.type or "other").lower()
        acct = h.account_name or "Unspecified"
        alloc_by_type[t] = alloc_by_type.get(t, 0.0) + val
        alloc_by_account[acct] = alloc_by_account.get(acct, 0.0) + val

    # 5) Connections (sync DB read is fine)
    connections = get_connections(user_id, db)

    # 6) Return (round ONLY at the edge)
    return {
        "as_of": enriched.get("as_of", int(time.time())),
        "requested_currency": currency.upper(),
        "price_status": price_status,
        "positions_count": len(items),
        "market_value": round(market_value, 2),
        "cost_basis": round(cost_basis, 2),
        "unrealized_pl": round(unrealized_pl, 2) if cost_basis > 0 else None,
        "unrealized_pl_pct": None if unrealized_pl_pct is None else round(unrealized_pl_pct, 2),
        "day_pl": None if prev_close_total == 0 else round(day_pl, 2),
        "day_pl_pct": None if day_pl_pct is None else round(day_pl_pct, 2),
        "allocations": {
            "by_type": _normalize_alloc(alloc_by_type, market_value),
            "by_account": _normalize_alloc(alloc_by_account, market_value),
        },
        "top_positions": enriched.get("top_items", []),
        "connections": connections,
    }
