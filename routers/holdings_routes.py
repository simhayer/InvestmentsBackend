# routers/holdings_routes.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import schemas.general as general
from database import get_db
from models.holding import Holding
from routers.finnhub_routes import get_finnhub_service
from services.finnhub_service import FinnhubService
from services.holding_service import get_all_holdings, get_holdings_with_live_prices, create_holding
from services.supabase_auth import get_current_db_user
from services.currency_service import resolve_currency

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
    )

@router.get("/holdings")
async def get_holdings(
    includePrices: bool = Query(False),
    currency: str | None = Query(None),  # None => use user base currency
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    resolved_currency = resolve_currency(user, currency)

    if not includePrices:
        # NOTE: this returns DB rows; if your frontend expects a consistent response shape, wrap it
        return get_all_holdings(str(user.id), db)

    return await get_holdings_with_live_prices(
        str(user.id),
        db,
        finnhub,
        currency=resolved_currency,
    )

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
