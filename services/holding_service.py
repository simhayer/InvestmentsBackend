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
    type_: str,
    name: str | None = None,
    currency: str = "USD",
) -> Holding:
    total_cost = purchase_price * quantity
    current_value = purchase_price * quantity  # initial estimate until live price

    holding = Holding(
        symbol=symbol.upper().strip(),
        name=name or symbol.upper().strip(),
        quantity=quantity,
        purchase_price=purchase_price,
        purchase_unit_price=purchase_price,
        purchase_amount_total=total_cost,
        current_price=purchase_price,  # placeholder until live enrichment
        current_value=current_value,
        value=current_value,
        unrealized_pl=0.0,
        unrealized_pl_pct=0.0,
        type=type_,
        currency=currency.upper(),
        source="manual",
        external_id=f"manual_{symbol.upper().strip()}_{int(time.time())}",
        institution="Manual",
        account_name="Manual",
        user_id=user_id,
    )
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return holding


def update_holding(
    db: Session,
    user_id: int,
    holding_id: int,
    updates: dict,
) -> Holding:
    """Update a manual holding. Only source='manual' holdings can be edited."""
    holding = db.query(Holding).filter_by(id=holding_id, user_id=user_id).first()
    if not holding:
        raise ValueError("Holding not found")
    if holding.source != "manual":
        raise PermissionError("Only manually added holdings can be edited")

    if "symbol" in updates and updates["symbol"] is not None:
        holding.symbol = updates["symbol"].upper().strip()
    if "name" in updates and updates["name"] is not None:
        holding.name = updates["name"]
    if "quantity" in updates and updates["quantity"] is not None:
        holding.quantity = updates["quantity"]
    if "purchase_price" in updates and updates["purchase_price"] is not None:
        holding.purchase_price = updates["purchase_price"]
        holding.purchase_unit_price = updates["purchase_price"]
    if "type" in updates and updates["type"] is not None:
        holding.type = updates["type"]
    if "currency" in updates and updates["currency"] is not None:
        holding.currency = updates["currency"].upper()

    # Recompute derived fields
    qty = holding.quantity or 0
    unit_price = holding.purchase_price or 0
    holding.purchase_amount_total = unit_price * qty
    holding.value = (holding.current_price or unit_price) * qty
    holding.current_value = holding.value

    if holding.purchase_amount_total and holding.purchase_amount_total > 0:
        holding.unrealized_pl = holding.current_value - holding.purchase_amount_total
        holding.unrealized_pl_pct = ((holding.current_value / holding.purchase_amount_total) - 1.0) * 100.0
    else:
        holding.unrealized_pl = 0.0
        holding.unrealized_pl_pct = 0.0

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


def get_holdings_broker_only(
    user_id: str,
    db: Session,
    currency: str | None = None,
) -> Dict[str, Any]:
    """
    Return holdings using broker (DB) as source of truth only: currency from broker,
    purchase_price, quantity, value = purchase_price * quantity. No live prices, no P/L.
    """
    rows = get_all_holdings(user_id, db)
    as_of = int(time.time())
    if not rows:
        return {"items": [], "top_items": [], "as_of": as_of, "currency": None}

    items: List[HoldingOut] = []
    for h in rows:
        qty = to_float(h.quantity)
        unit_price = to_float(getattr(h, "purchase_unit_price", None)) or to_float(h.purchase_price)
        # Value = purchase_price * quantity (cost basis); no current price
        value = round(unit_price * qty, 8) if unit_price and qty else 0.0
        dto = HoldingOut(
            id=h.id,
            user_id=h.user_id,
            source=h.source,
            external_id=h.external_id,
            symbol=h.symbol,
            name=h.name,
            type=h.type,
            quantity=h.quantity,
            purchase_price=h.purchase_price,
            value=value,
            account_name=h.account_name,
            institution=h.institution,
            currency=(h.currency or "USD").strip().upper(),
            purchase_amount_total=getattr(h, "purchase_amount_total", None),
            purchase_unit_price=getattr(h, "purchase_unit_price", None),
            current_price=None,
            current_value=value,
            unrealized_pl=None,
            unrealized_pl_pct=None,
            previous_close=None,
            price_status=None,
            day_pl=None,
            weight=None,
        )
        items.append(dto)

    # Optional: compute weights by cost (value)
    total = sum(to_float(it.value) for it in items)
    for it in items:
        it.weight = round(to_float(it.value) / total * 100.0, 8) if total > 0 else 0.0

    top_items = sorted(items, key=lambda x: to_float(x.value), reverse=True)[:5]
    return {
        "items": items,
        "top_items": top_items,
        "market_value": total,
        "as_of": as_of,
        "currency": None,  # per-holding currency from broker
    }


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
    """
    Build DTOs for holdings. Prefer brokerage (Plaid) data when available so
    current price, value, and P/L stay consistent with the institution. Only
    use Finnhub for manual holdings or when brokerage did not provide a price.
    """
    currency_default = currency.upper()
    enriched: List[HoldingOut] = []

    for h in holdings:
        dto = _base_dto_from_row(h, currency_default)
        typ_n = normalize_asset_type(h.type)
        k_full = canonical_key(h.symbol, typ_n)
        q = quotes.get(k_full)
        qty = to_float(h.quantity)

        # Prefer brokerage data for Plaid holdings: use DB current_price, value, P/L
        # so maths match the institution (avg cost, P/L, etc.) instead of Finnhub.
        db_price = to_float(h.current_price)
        db_value = to_float(getattr(h, "current_value", None)) or to_float(h.value)
        from_brokerage = (
            getattr(h, "source", None) == "plaid"
            and db_price is not None
            and db_price > 0
        )

        if from_brokerage:
            dto.current_price = db_price
            if db_value is not None and db_value > 0:
                dto.value = round(db_value, 8)
            else:
                dto.value = round(db_price * qty, 8) if qty > 0 else 0.0
            dto.current_value = dto.value
            dto.price_status = "brokerage"
        elif q and q.get("currentPrice") is not None:
            live_price = to_float(q["currentPrice"])
            dto.current_price = live_price
            dto.previous_close = to_float(q.get("previousClose")) if q.get("previousClose") is not None else None
            dto.currency = (q.get("currency") or dto.currency or currency_default).upper()
            dto.price_status = "live"
            price = live_price
            dto.value = round(price * qty, 8) if price > 0 and qty > 0 else max(to_float(dto.value), 0.0)
            dto.current_value = dto.value
        else:
            dto.price_status = "unavailable"
            price = to_float(dto.current_price)
            if price > 0 and qty > 0:
                dto.value = round(price * qty, 8)
            else:
                unit_cost = to_float(getattr(h, "purchase_unit_price", None)) or to_float(h.purchase_price)
                if unit_cost and unit_cost > 0 and qty > 0:
                    dto.current_price = unit_cost
                    dto.value = round(unit_cost * qty, 8)
                else:
                    dto.value = max(to_float(dto.value), 0.0)
            dto.current_value = dto.value

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
        # Get the original database row to check its specific currency (from broker via Plaid).
        # Do not default brokerage holdings to USD when currency is missing—that causes CAD
        # stocks (e.g. FLT, VFV) to be double-converted. Default plaid to CAD when empty.
        original_row = rows_map.get(it.id)
        raw_currency = getattr(original_row, "currency", None) or ""
        raw_currency = (raw_currency or "").strip().upper()
        if not raw_currency and getattr(original_row, "source", None) == "plaid":
            raw_currency = "CAD"
        db_currency = (raw_currency or "USD").upper()
        qty = to_float(it.quantity)
        is_brokerage = getattr(it, "price_status", None) == "brokerage"

        # --- STEP 1: Convert current price (and value for brokerage) to target currency ---
        # Finnhub is always USD. Brokerage data uses db_currency.
        if is_brokerage:
            if db_currency == "USD" and target_curr == "CAD":
                it.current_price = round(to_float(it.current_price) * usd_to_cad_rate, 8)
                it.value = round(to_float(it.value) * usd_to_cad_rate, 8)
                it.current_value = it.value
            elif db_currency == "CAD" and target_curr == "USD":
                it.current_price = round(to_float(it.current_price) / usd_to_cad_rate, 8)
                it.value = round(to_float(it.value) / usd_to_cad_rate, 8)
                it.current_value = it.value
        else:
            if target_curr == "CAD":
                it.current_price = round(to_float(it.current_price) * usd_to_cad_rate, 8)
                if it.previous_close:
                    it.previous_close = round(it.previous_close * usd_to_cad_rate, 8)

        # --- STEP 2: Handle Cost Basis (DB Currency -> Target) ---
        if db_currency == "CAD" and target_curr == "USD":
            it.purchase_unit_price = round(to_float(it.purchase_unit_price) / usd_to_cad_rate, 8)
            it.purchase_amount_total = round(to_float(it.purchase_amount_total) / usd_to_cad_rate, 8)
        elif db_currency == "USD" and target_curr == "CAD":
            it.purchase_unit_price = round(to_float(it.purchase_unit_price) * usd_to_cad_rate, 8)
            it.purchase_amount_total = round(to_float(it.purchase_amount_total) * usd_to_cad_rate, 8)

        # --- STEP 3: Finalize value (brokerage already set in STEP 1) ---
        it.currency = target_curr
        if not is_brokerage:
            calc_val = round(to_float(it.current_price) * qty, 8)
            it.value = calc_val
            it.current_value = calc_val
        else:
            calc_val = to_float(it.value)

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