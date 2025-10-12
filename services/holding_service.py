from decimal import Decimal
import time
from sqlalchemy.orm import Session
from services.finnhub_service import FinnhubService
from models.holding import Holding
from typing import Any, Dict, List, Tuple

def _to_float(x: Any) -> float:
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return 0.0

def _position_value(h: Dict[str, Any]) -> float:
    v = _to_float(h.get("value"))
    if v > 0:
        return v
    return _to_float(h.get("current_price")) * _to_float(h.get("quantity"))

def _enrich_pl_fields(h: Dict[str, Any]) -> Dict[str, Any]:
    # returns a copy with day_pl / unrealized_pl
    out = dict(h)
    qty  = _to_float(h.get("quantity"))
    curr = _to_float(h.get("current_price"))
    pur  = _to_float(h.get("purchase_price"))
    pc   = _to_float(h.get("previous_close")) if "previous_close" in h else _to_float(h.get("previousClose"))
    out["day_pl"] = None if pc <= 0 else round((curr - pc) * qty, 2)
    out["unrealized_pl"] = None if pur <= 0 else round((curr - pur) * qty, 2)
    return out

def get_all_holdings(user_id: str, db: Session) -> list[dict[str, Any]]:
    """
    Get all holdings for a user.
    """
    holdings = db.query(Holding).filter_by(user_id=user_id).all()

    return [
        {
            "id": h.id,
            "symbol": h.symbol,
            "name": h.name,
            "type": h.type,
            "quantity": h.quantity,
            "purchase_price": h.purchase_price,
            "current_price": h.current_price,
            "value": h.value,
            "currency": h.currency,
            "institution": h.institution,
            "account_name": h.account_name,
            "source": h.source,
        }
        for h in holdings
    ]

async def get_holdings_with_live_prices(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
) -> Dict[str, Any]:
    """
    Returns holdings enriched with live prices from Finnhub.
    Never mutates DB; this is a read-time aggregation.
    """
    rows = get_all_holdings(user_id, db)

    if not rows:
        return {"items": [], "as_of": int(time.time()), "price_status": "live"}

    # De-dupe by (symbol, type) because the same security can appear across accounts.
    pairs: List[Tuple[str, str]] = list({(h["symbol"], h["type"]) for h in rows})
    quotes = await finnhub.get_prices(pairs=pairs, currency=currency)

    items: List[Dict[str, Any]] = []
    for h in rows:
        q = quotes.get(h["symbol"])

        # Convert Decimal -> float for arithmetic if your model uses Decimal
        qty = float(h["quantity"]) if isinstance(h["quantity"], Decimal) else (h["quantity"] or 0.0)

        enriched = dict(h)  # shallow copy
        if q and q.get("currentPrice"):
            live_price = float(q["currentPrice"])
            enriched["current_price"] = live_price
            enriched["previous_close"] = float(q.get("previousClose") or 0)
            enriched["currency"] = q.get("currency", currency.upper())
            enriched["value"] = round(live_price * qty, 2)
            enriched["price_status"] = "live"
        else:
            # Keep whatever you had in DB, but mark as unavailable
            enriched["price_status"] = "unavailable"

            # If DB has price, recompute value to be consistent
            if enriched.get("current_price") is not None:
                try:
                    enriched["value"] = round(float(enriched["current_price"]) * qty, 2)
                except Exception:
                    pass

        items.append(enriched)

    return {
        "items": items,
        "as_of": int(time.time()),
        "price_status": "live",
        "requested_currency": currency.upper(),
    }

async def get_holdings_with_live_prices_top(
    user_id: str,
    db,
    finnhub,
    currency: str = "USD",
    *,
    top_only: bool = False,
    top_n: int = 5,
    include_weights: bool = True,
) -> Dict[str, Any]:
    """
    Returns holdings enriched with live prices from Finnhub.
    If top_only=True, returns the top N positions by value but keeps the SAME shape.
    Never mutates DB.
    """
    rows = get_all_holdings(user_id, db)
    if not rows:
        return {"items": [], "as_of": int(time.time()), "price_status": "live", "requested_currency": currency.upper()}

    # De-dupe by (symbol, type) because the same security can appear across accounts.
    pairs: List[Tuple[str, str]] = list({(h["symbol"], h["type"]) for h in rows})
    quotes = await finnhub.get_prices(pairs=pairs, currency=currency)

    items: List[Dict[str, Any]] = []
    for h in rows:
        q = quotes.get(h["symbol"])
        qty = _to_float(h.get("quantity") or 0.0)

        enriched = dict(h)  # shallow copy
        if q and q.get("currentPrice"):
            live_price = _to_float(q["currentPrice"])
            enriched["current_price"] = live_price
            enriched["previous_close"] = _to_float(q.get("previousClose") or 0)
            enriched["currency"] = q.get("currency", currency.upper())
            enriched["value"] = round(live_price * qty, 2)
            enriched["price_status"] = "live"
        else:
            enriched["price_status"] = "unavailable"
            # If DB had a price, recompute value consistently
            if enriched.get("current_price") is not None:
                try:
                    enriched["value"] = round(_to_float(enriched["current_price"]) * qty, 2)
                except Exception:
                    pass

        items.append(enriched)

    # If not requesting top, return as-is (original behavior)
    payload = {
        "items": items,
        "as_of": int(time.time()),
        "price_status": "live",
        "requested_currency": currency.upper(),
    }
    if not top_only or not items:
        return payload

    # Compute market value to calculate weights
    market_value = 0.0
    for it in items:
        v = _position_value(it)
        market_value += max(v, 0.0)

    # Pick top N by value; keep SAME item shape and just add optional fields
    sorted_items = sorted(items, key=_position_value, reverse=True)
    top_items: List[Dict[str, Any]] = []
    for it in sorted_items[: max(1, top_n)]:
        v = _position_value(it)
        copy = _enrich_pl_fields(it)
        copy["value"] = round(v, 2)  # ensure consistent value
        if include_weights:
            copy["weight"] = round(v / market_value * 100.0, 2) if market_value > 0 else None
        top_items.append(copy)

    return {**payload, "items": top_items}