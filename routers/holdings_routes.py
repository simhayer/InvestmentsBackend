# routers/holdings_routes.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import schemas.general as general
from database import get_db
from models.holding import Holding
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService, format_finnhub_symbol
from services.holding_service import get_all_holdings, get_holdings_broker_only, create_holding, update_holding
from services.currency_service import resolve_currency, fx_pair_rate
from services.supabase_auth import get_current_db_user
from utils.common_helpers import to_float, canonical_key, normalize_asset_type

router = APIRouter()

@router.post("/holdings")
def save_holding(
    holding: general.HoldingCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    return create_holding(
        db,
        user.id,
        holding.symbol,
        holding.quantity,
        holding.purchase_price,
        holding.type,
        name=holding.name,
        currency=holding.currency or "USD",
    )

@router.put("/holdings/{holding_id}")
def edit_holding(
    holding_id: int,
    payload: general.HoldingUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    try:
        updates = payload.model_dump(exclude_none=True)
        updated = update_holding(db, user.id, holding_id, updates)
        return updated
    except ValueError:
        raise HTTPException(status_code=404, detail="Holding not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Only manually added holdings can be edited")

@router.get("/holdings")
async def get_holdings(
    includePrices: bool = Query(False),
    currency: str | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """Holdings from broker (DB) only. Total cost is converted to the user's chosen currency (settings).
    When includePrices=true, current_price is filled from Finnhub (Canadian equity/ETF use TSX .TO for CAD;
    crypto quotes are in USD and are converted to the holding's currency when different)."""
    if not includePrices:
        return get_all_holdings(str(user.id), db)

    data = get_holdings_broker_only(str(user.id), db)
    target_currency = resolve_currency(user, currency)
    if not data.get("items"):
        data["currency"] = target_currency
        return data

    # Build (symbol, type) pairs for Finnhub; use TSX .TO for CAD equity/ETF so prices are in CAD
    pairs: list[tuple[str, str]] = []
    for it in data["items"]:
        sym = (getattr(it, "symbol", None) or "").strip()
        typ = getattr(it, "type", None) or ""
        ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        if not sym:
            continue
        fs = format_finnhub_symbol(sym, typ, ccy)
        typ_n = normalize_asset_type(typ) or "equity"
        pairs.append((fs, typ_n))

    if pairs:
        quotes = await finnhub.get_prices_cached(pairs)
        for it in data["items"]:
            sym = (getattr(it, "symbol", None) or "").strip()
            typ = getattr(it, "type", None) or ""
            ccy = (getattr(it, "currency", None) or "USD").strip().upper()
            if not sym:
                continue
            fs = format_finnhub_symbol(sym, typ, ccy)
            typ_n = normalize_asset_type(typ) or "equity"
            key = canonical_key(fs, typ_n)
            q = quotes.get(key)
            if isinstance(q, dict) and q.get("currentPrice") is not None:
                price = float(q["currentPrice"])
                # Finnhub crypto quotes are always in USD; convert to holding currency if needed
                if typ_n == "cryptocurrency" and ccy != "USD":
                    rate = await fx_pair_rate("USD", ccy)
                    price = round(price * rate, 8)
                it.current_price = price

    total_in_target = 0.0
    for it in data["items"]:
        from_ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        rate = await fx_pair_rate(from_ccy, target_currency)
        total_in_target += to_float(getattr(it, "value", 0)) * rate
    data["market_value"] = round(total_in_target, 2)
    data["currency"] = target_currency
    return data

@router.delete("/holdings/{holding_id}")
def delete_holding( 
    holding_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    holding = db.query(Holding).filter_by(id=holding_id, user_id=user.id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")

    db.delete(holding)
    db.commit()
    return {"detail": "Deleted"}
