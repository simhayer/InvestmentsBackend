# services/portfolio_service.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from decimal import Decimal
import math, time

from sqlalchemy.orm import Session
from services.finnhub_service import FinnhubService
from services.holding_service import get_all_holdings
# If you already have get_holdings_with_live_prices, we’ll reuse it directly:
from services.holding_service import get_holdings_with_live_prices

Number = float | int | Decimal

def _to_float(x: Number | None) -> float:
    if x is None:
        return 0.0
    if isinstance(x, Decimal):
        return float(x)
    return float(x)

async def get_portfolio_summary(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    top_n: int = 5,
) -> Dict[str, Any]:
    # Enrich holdings with live quotes (current_price, currency, value, ...).
    enriched = await get_holdings_with_live_prices(user_id, db, finnhub, currency=currency)
    items: List[Dict[str, Any]] = enriched.get("items", [])

    # Totals
    market_value = 0.0
    cost_basis = 0.0
    day_pl = 0.0
    prev_close_total = 0.0

    for h in items:
        qty = _to_float(h.get("quantity"))
        curr = _to_float(h.get("current_price"))
        pur = _to_float(h.get("purchase_price"))
        pc  = _to_float(h.get("previous_close")) if "previous_close" in h else _to_float(h.get("previousClose"))

        market_value += curr * qty
        if pur > 0:
            cost_basis += pur * qty
        if pc > 0:
            day_pl += (curr - pc) * qty
            prev_close_total += pc * qty

    unrealized_pl = market_value - cost_basis if cost_basis > 0 else 0.0
    unrealized_pl_pct = (unrealized_pl / cost_basis * 100.0) if cost_basis > 0 else None
    day_pl_pct = (day_pl / prev_close_total * 100.0) if prev_close_total > 0 else None

    # Allocations
    def add_alloc(bucket: Dict[str, float], key: str, value: float):
        bucket[key] = bucket.get(key, 0.0) + value

    alloc_by_type: Dict[str, float] = {}
    alloc_by_account: Dict[str, float] = {}

    for h in items:
        val = _to_float(h.get("value"))
        if val <= 0:
            qty = _to_float(h.get("quantity"))
            curr = _to_float(h.get("current_price"))
            val = curr * qty
        add_alloc(alloc_by_type, (h.get("type") or "other").lower(), val)
        add_alloc(alloc_by_account, h.get("account_name") or "Unspecified", val)

    def normalize_alloc(d: Dict[str, float]) -> List[Dict[str, Any]]:
        if market_value <= 0:
            return [{"key": k, "value": round(v, 2), "weight": None} for k, v in sorted(d.items(), key=lambda x: -x[1])]
        return [
            {"key": k, "value": round(v, 2), "weight": round(v / market_value * 100.0, 2)}
            for k, v in sorted(d.items(), key=lambda x: -x[1])
        ]

    # Top positions by weight
    def position_value(h: Dict[str, Any]) -> float:
        v = _to_float(h.get("value"))
        if v > 0:
            return v
        return _to_float(h.get("current_price")) * _to_float(h.get("quantity"))

    sorted_positions = sorted(items, key=position_value, reverse=True)
    top_positions = []
    for h in sorted_positions[:top_n]:
        v = position_value(h)
        qty = _to_float(h.get("quantity"))
        curr = _to_float(h.get("current_price"))
        pur = _to_float(h.get("purchase_price"))
        pc  = _to_float(h.get("previous_close")) if "previous_close" in h else _to_float(h.get("previousClose"))
        unreal = (curr - pur) * qty if pur > 0 else None
        dayp   = (curr - pc) * qty if pc > 0 else None
        top_positions.append({
            "symbol": h.get("symbol"),
            "name": h.get("name"),
            "type": h.get("type"),
            "value": round(v, 2),
            "weight": round(v / market_value * 100.0, 2) if market_value > 0 else None,
            "unrealized_pl": None if unreal is None else round(unreal, 2),
            "day_pl": None if dayp is None else round(dayp, 2),
        })

    return {
        "as_of": enriched.get("as_of", int(time.time())),
        "currency": currency.upper(),
        "positions_count": len(items),
        "market_value": round(market_value, 2),
        "cost_basis": round(cost_basis, 2),
        "unrealized_pl": round(unrealized_pl, 2) if cost_basis > 0 else None,
        "unrealized_pl_pct": None if unrealized_pl_pct is None else round(unrealized_pl_pct, 2),
        "day_pl": None if prev_close_total == 0 else round(day_pl, 2),
        "day_pl_pct": None if day_pl_pct is None else round(day_pl_pct, 2),
        "allocations": {
            "by_type": normalize_alloc(alloc_by_type),
            "by_account": normalize_alloc(alloc_by_account),
        },
        "top_positions": top_positions,
    }
