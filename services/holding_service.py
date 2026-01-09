from __future__ import annotations
import time
import heapq
from typing import Any, Dict, List, Tuple
from sqlalchemy.orm import Session
from models.holding import Holding, HoldingOut
from services.finnhub.finnhub_service import FinnhubService
from utils.common_helpers import to_float
from utils.common_helpers import canonical_key, normalize_asset_type
from services.currency_service import get_usd_to_cad_rate

def create_holding(
    db: Session,
    user_id: int,
    symbol: str,
    quantity: float,
    purchase_price: float,
    type_: str
) -> Holding:
    holding = Holding(
        symbol=symbol,
        quantity=quantity,
        purchase_price=purchase_price,
        type=type_,
        user_id=user_id,
    )
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return holding

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
        typ_n = normalize_asset_type(h.type)
        k_full = canonical_key(h.symbol, typ_n)
        q = quotes.get(k_full)

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
    # 1. Fetch rows (Holding objects with .currency attribute)
    rows = get_all_holdings(user_id, db)
    as_of = int(time.time())
    target_curr = currency.upper()

    if not rows:
        return {"items": [], "top_items": [], "as_of": as_of, "currency": target_curr}

    # 2. Map rows for easy lookup during enrichment
    rows_map = {h.id: h for h in rows}
    
    # 3. Fetch raw prices (Finnhub typically returns USD for US stocks/crypto)
    pairs = [((h.symbol or "").strip(), normalize_asset_type(h.type) or "equity") for h in rows]
    quotes = await finnhub.get_prices_cached(pairs=pairs, currency="USD")
    
    usd_to_cad_rate = await get_usd_to_cad_rate()
    enriched_items = enrich_holdings(rows, quotes, "USD")
    
    market_value = 0.0
    valued: List[Tuple[float, HoldingOut]] = []

    for it in enriched_items:
        # Get the original database row to check its specific currency
        original_row = rows_map.get(it.id)
        db_currency = (getattr(original_row, "currency", "USD") or "USD").upper()
        qty = to_float(it.quantity)

        # --- STEP 1: Handle Live Price (Finnhub USD -> Target) ---
        # Finnhub results are in USD. If target is CAD, convert.
        if target_curr == "CAD":
            it.current_price = round(to_float(it.current_price) * usd_to_cad_rate, 8)
            if it.previous_close:
                it.previous_close = round(it.previous_close * usd_to_cad_rate, 8)
        elif target_curr == "USD":
            # Already in USD from Finnhub, no conversion needed
            pass

        # --- STEP 2: Handle Cost Basis (DB Currency -> Target) ---
        # If the DB stores CAD but we want USD portfolio view
        if db_currency == "CAD" and target_curr == "USD":
            it.purchase_unit_price = round(to_float(it.purchase_unit_price) / usd_to_cad_rate, 8)
            it.purchase_amount_total = round(to_float(it.purchase_amount_total) / usd_to_cad_rate, 8)
        
        # If the DB stores USD but we want CAD portfolio view
        elif db_currency == "USD" and target_curr == "CAD":
            it.purchase_unit_price = round(to_float(it.purchase_unit_price) * usd_to_cad_rate, 8)
            it.purchase_amount_total = round(to_float(it.purchase_amount_total) * usd_to_cad_rate, 8)

        # --- STEP 3: Finalize Values ---
        it.currency = target_curr
        calc_val = round(to_float(it.current_price) * qty, 8)
        
        it.value = calc_val
        it.current_value = calc_val
        
        market_value += calc_val
        valued.append((calc_val, it))

    # 4. Weights and P/L
    for v, it in valued:
        it.weight = round(v / market_value * 100.0, 8) if market_value > 0 else 0.0
        _compute_pl_fields(it)

    top_items = [it for _, it in heapq.nlargest(top_n, valued, key=lambda t: t[0])]

    return {
        "items": top_items if top_only else enriched_items,
        "top_items": top_items,
        "market_value": market_value,
        "as_of": as_of,
        "currency": target_curr,
    }