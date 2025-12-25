from __future__ import annotations
import time
import heapq
from typing import Any, Dict, List, Tuple
from sqlalchemy.orm import Session
from models.holding import Holding, HoldingOut
from services.finnhub_service import FinnhubService
from utils.common_helpers import to_float
from utils.common_helpers import canonical_key
from services.currency_service import get_usd_to_cad_rate

def _compute_pl_fields(h: HoldingOut) -> None:
    qty = to_float(getattr(h, "quantity", 0.0))

    curr = getattr(h, "current_price", None)
    pc = getattr(h, "previous_close", None)

    total_cost = getattr(h, "purchase_amount_total", None)
    unit_cost = getattr(h, "purchase_unit_price", None) or getattr(h, "purchase_price", None)
    value = getattr(h, "value", None)

    curr_f = to_float(curr) if curr is not None else None
    pc_f = to_float(pc) if pc is not None else None
    total_cost_f = to_float(total_cost) if total_cost is not None else None
    unit_cost_f = to_float(unit_cost) if unit_cost is not None else None
    value_f = to_float(value) if value is not None else None

    # Day P/L
    if curr_f is not None and pc_f is not None and pc_f > 0 and qty > 0:
        h.day_pl = round((curr_f - pc_f) * qty, 8)
    else:
        h.day_pl = None

    # Unrealized P/L
    if total_cost_f is not None and total_cost_f > 0 and value_f is not None:
        pl = value_f - total_cost_f
        h.unrealized_pl = round(pl, 8)
        h.unrealized_pl_pct = round((value_f / total_cost_f - 1.0) * 100.0, 8)
    elif curr_f is not None and unit_cost_f is not None and unit_cost_f > 0 and qty > 0:
        pl = (curr_f - unit_cost_f) * qty
        h.unrealized_pl = round(pl, 8)
        h.unrealized_pl_pct = round((curr_f / unit_cost_f - 1.0) * 100.0, 8)
    else:
        h.unrealized_pl = None
        h.unrealized_pl_pct = None

def get_all_holdings(user_id: str, db: Session) -> List[Holding]:
    return db.query(Holding).filter_by(user_id=user_id).all()

# -----------------------
# Mapping + enrichment
# -----------------------

def _base_dto_from_row(h: Holding, currency_default: str) -> HoldingOut:
    return HoldingOut(
        id=h.id,
        user_id=h.user_id,
        source=h.source,
        external_id=h.external_id,
        symbol=h.symbol,
        name=h.name,
        type=h.type,
        quantity=h.quantity,
        current_price=h.current_price,
        purchase_price=h.purchase_price,
        value=h.value,
        account_name=h.account_name,
        institution=h.institution,
        currency=(h.currency or currency_default),
        purchase_amount_total=getattr(h, "purchase_amount_total", None),
        purchase_unit_price=getattr(h, "purchase_unit_price", None),
        unrealized_pl=getattr(h, "unrealized_pl", None),
        unrealized_pl_pct=getattr(h, "unrealized_pl_pct", None),
        current_value=getattr(h, "current_value", None),
    )


def enrich_holdings(
    holdings: List[Holding],
    quotes: Dict[str, Dict[str, Any] | None],
    currency: str = "USD",
) -> List[HoldingOut]:
    currency_default = currency.upper()
    enriched: List[HoldingOut] = []

    for h in holdings:
        dto = _base_dto_from_row(h, currency_default)

        k_full = canonical_key(h.symbol, h.type)
        k_sym = canonical_key(h.symbol, None)
        q = quotes.get(k_full) or quotes.get(k_sym)

        qty = to_float(h.quantity)

        if q and q.get("currentPrice") is not None:
            live_price = to_float(q["currentPrice"])
            dto.current_price = live_price
            dto.previous_close = to_float(q.get("previousClose")) if q.get("previousClose") is not None else None
            dto.currency = (q.get("currency") or dto.currency or currency_default).upper()
            dto.price_status = "live"
        else:
            dto.price_status = "unavailable"

        # set value consistently for everyone (live OR stored price)
        price = to_float(dto.current_price)
        dto.value = round(price * qty, 8) if price > 0 and qty > 0 else max(to_float(dto.value), 0.0)

        enriched.append(dto)

    return enriched

async def get_holdings_with_live_prices(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    *,
    top_only: bool = False,
    top_n: int = 5,
    include_weights: bool = True,
) -> Dict[str, Any]:
    rows = get_all_holdings(user_id, db)
    as_of = int(time.time())
    currency_up = currency.upper()

    if not rows:
        return {
            "items": [],
            "top_items": [],
            "as_of": as_of,
            "price_status": "live",
            "currency": currency_up,
        }

    # request pairs
    pairs: List[Tuple[str, str]] = [
        ((h.symbol or "").strip(), (h.type or "").strip())
        for h in rows
        if (h.symbol or "").strip()
    ]

    quotes_raw = await finnhub.get_prices(pairs=pairs, currency=currency_up)
    quotes = quotes_raw or {}
    if currency_up == "CAD":
        usd_to_cad = await get_usd_to_cad_rate()

        for k, q in quotes.items():
            if not q:
                continue

            def conv(x):
                if x is None:
                    return None
                try:
                    return round(float(x) * usd_to_cad, 8)
                except Exception:
                    return None

            q["currentPrice"] = conv(q.get("currentPrice"))
            q["previousClose"] = conv(q.get("previousClose"))
            q["high"] = conv(q.get("high"))
            q["low"] = conv(q.get("low"))
            q["open"] = conv(q.get("open"))
            q["currency"] = "CAD"
    items = enrich_holdings(rows, quotes, currency_up)

    # compute position values ONCE and reuse everywhere
    valued: List[Tuple[float, HoldingOut]] = []
    market_value = 0.0

    for it in items:
        v = max(to_float(getattr(it, "value", 0.0) or getattr(it, "current_value", 0.0)), 0.0)

        # fallback if value wasn’t present for some reason
        if v <= 0:
            v = max(to_float(getattr(it, "current_price", 0.0)) * to_float(getattr(it, "quantity", 0.0)), 0.0)
        v = round(v, 8)
        it.value = v
        it.current_value = v

        market_value += v
        valued.append((v, it))

    # derived fields (weight + P/L) using cached values
    for v, it in valued:
        if include_weights and market_value > 0 and v > 0:
            it.weight = round(v / market_value * 100.0, 8)
        else:
            it.weight = None

        _compute_pl_fields(it)

    # top items WITHOUT sorting the full list
    n = max(1, top_n)
    top_items = [it for _, it in heapq.nlargest(n, valued, key=lambda t: t[0])]

    return {
        "items": top_items if top_only else items,
        "top_items": top_items,
        "as_of": as_of,
        "price_status": "live",
        "currency": currency_up,
    }
