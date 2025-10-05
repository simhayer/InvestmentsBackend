from decimal import Decimal
import time
from sqlalchemy.orm import Session
from services.finnhub_service import FinnhubService
from models.holding import Holding
from typing import Any, Dict, List, Tuple

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