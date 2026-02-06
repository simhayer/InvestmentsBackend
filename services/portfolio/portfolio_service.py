# services/portfolio_service.py
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from math import fsum
from typing import Any, Dict, List
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from models.holding import HoldingOut
from models.portfolio_analysis import PortfolioAnalysis
from services.finnhub.finnhub_service import FinnhubService
from services.holding_service import get_holdings_with_live_prices
from services.plaid.plaid_service import get_connections
from utils.common_helpers import to_float

Number = float | int | Decimal

def _normalize_alloc(d: Dict[str, float], total: float) -> List[Dict[str, Any]]:
    items = sorted(d.items(), key=lambda kv: -kv[1])
    if total <= 0:
        return [{"key": k, "value": round(v,8), "weight": None} for k, v in items]
    return [{"key": k, "value": round(v, 8), "weight": round(v / total * 100.0, 8)} for k, v in items]

async def get_portfolio_summary(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    top_n: int = 5,
    holdings_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Aggregates portfolio summary from holdings enriched with live prices.
    """
    if holdings_payload is None:
        enriched = await get_holdings_with_live_prices(
            user_id,
            db,
            finnhub,
            currency=currency,
            top_only=False,
            top_n=top_n,
            include_weights=True,
        )
    else:
        enriched = holdings_payload

    items: List[HoldingOut] = enriched.get("items", [])
    # Route-level price status (live/mixed/unavailable)
    live_count = sum(1 for it in items if getattr(it, "price_status", None) == "live")
    if not items or live_count == 0:
        price_status = "unavailable"
    elif live_count == len(items):
        price_status = "live"
    else:
        price_status = "mixed"

    # Totals: rely on computed holding values
    values = [to_float(getattr(h, "value", 0.0)) for h in items]
    market_value = fsum(values)
    top_positions = enriched.get("top_items", [])

    # Cost basis: prefer purchase_amount_total if present, else unit * qty
    cost_terms: List[float] = []
    for h in items:
        total_cost = to_float(getattr(h, "purchase_amount_total", None))
        if total_cost > 0:
            cost_terms.append(total_cost)
            continue

        qty = to_float(getattr(h, "quantity", 0.0))
        unit = to_float(getattr(h, "purchase_unit_price", None) or getattr(h, "purchase_price", None))
        if qty > 0 and unit > 0:
            cost_terms.append(qty * unit)

    cost_basis = fsum(cost_terms)

    # P/L totals: sum the already computed values from holding_service to stay consistent
    unreal_terms = [
        to_float(getattr(h, "unrealized_pl", None))
        for h in items
        if getattr(h, "unrealized_pl", None) is not None
    ]
    day_terms = [
        to_float(getattr(h, "day_pl", None))
        for h in items
        if getattr(h, "day_pl", None) is not None
    ]
    unrealized_pl = fsum(unreal_terms) if unreal_terms else 0.0
    day_pl = fsum(day_terms) if day_terms else 0.0

    unrealized_pl_pct = (unrealized_pl / cost_basis * 100.0) if cost_basis > 0 else None

    # Day P/L % needs a comparable denominator (prev_close_total) in the same currency
    # holding_service sets previous_close along with current_price (same quote currency),
    # so this stays coherent.
    prev_close_total = fsum(
        to_float(getattr(h, "previous_close", 0.0)) * to_float(getattr(h, "quantity", 0.0))
        for h in items
        if getattr(h, "previous_close", None) not in (None, 0)
    )
    day_pl_pct = (day_pl / prev_close_total * 100.0) if prev_close_total > 0 else None

    # Allocations: use computed holding value
    alloc_by_type: Dict[str, float] = {}
    alloc_by_account: Dict[str, float] = {}

    for h in items:
        val = to_float(getattr(h, "value", 0.0))
        t = (h.type or "other").lower()
        acct = h.account_name or "Unspecified"
        alloc_by_type[t] = alloc_by_type.get(t, 0.0) + val
        alloc_by_account[acct] = alloc_by_account.get(acct, 0.0) + val

    connections = get_connections(user_id, db)

    return {
        "as_of": enriched.get("as_of", int(time.time())),
        "currency": currency.upper(),
        "price_status": price_status,
        "positions_count": len(items),
        "market_value": round(market_value, 8),
        "cost_basis": round(cost_basis, 8),
        "unrealized_pl": None if cost_basis <= 0 else round(unrealized_pl, 8),
        "unrealized_pl_pct": None if unrealized_pl_pct is None else round(unrealized_pl_pct, 8),
        "day_pl": None if prev_close_total <= 0 else round(day_pl, 8),
        "day_pl_pct": None if day_pl_pct is None else round(day_pl_pct, 8),
        "allocations": {
            "by_type": _normalize_alloc(alloc_by_type, market_value),
            "by_account": _normalize_alloc(alloc_by_account, market_value),
        },
        "top_positions": top_positions,
        "connections": connections,
    }

TTL_HOURS = 24

async def get_or_compute_portfolio_analysis(
    user_id: str,
    db: Session,
    *,
    base_currency: str = "USD",
    days_of_news: int = 7,
    targets: dict[str, int] | None = None,
    force: bool = False,
    finnhub: FinnhubService,
):
    now = datetime.now(timezone.utc)

    row = db.query(PortfolioAnalysis).filter(PortfolioAnalysis.user_id == user_id).first()
    if row and not force:
        age = now - row.created_at
        if age <= timedelta(hours=TTL_HOURS):
            rem = timedelta(hours=TTL_HOURS) - age
            meta = {
                "cached": True,
                "cached_at": row.created_at.isoformat(),
                "ttl_seconds_remaining": int(rem.total_seconds()),
            }
            return row.data, meta

    holdings_payload = await get_holdings_with_live_prices(
        user_id,
        db,
        currency=base_currency,
        top_only=False,
        top_n=0,
        include_weights=False,
        finnhub=finnhub,
    )

    items = (holdings_payload or {}).get("items") or []
    if not items:
        return None, {"reason": "no_holdings"}

    # FIX - removed for cleanup; re-add when pipeline is ready
    # ai_layers = await run_portfolio_pipeline(
    #     holdings_items=items,
    #     base_currency=base_currency,
    #     benchmark_ticker="SPY",
    #     days_of_news=days_of_news,
    # )

    data = {
        "version": "v1",
        "computed_at": now.isoformat(),
        "params": {
            "base_currency": base_currency,
            "days_of_news": days_of_news,
            "targets": targets,
        },
        "ai_layers": None,
    }

    stmt = insert(PortfolioAnalysis).values(user_id=user_id, data=data)
    upsert = stmt.on_conflict_do_update(
        index_elements=["user_id"],
        set_={"data": stmt.excluded.data, "created_at": func.now()},
    )
    db.execute(upsert)
    db.commit()

    meta = {
        "cached": False,
        "cached_at": now.isoformat(),
        "ttl_seconds_remaining": TTL_HOURS * 3600,
    }
    return data, meta
