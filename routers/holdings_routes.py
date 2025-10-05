# routers/holdings_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from routers.finnhub_routes import get_finnhub_service
from services.finnhub_service import FinnhubService
from database import get_db
from services.auth_service import get_current_user
import schemas.general as general, services.crud as crud
from models.holding import Holding
from services.holding_service import get_all_holdings, get_holdings_with_live_prices

router = APIRouter()

@router.post("/holdings")
def save_holding(holding: general.HoldingCreate, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.create_holding(db, user.id, holding.symbol, holding.quantity, holding.purchase_price, holding.type)

@router.get("/holdings")
async def get_holdings(
    includePrices: bool = Query(False),
    currency: str = Query("USD"),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    if not includePrices:
        return get_all_holdings(user.id, db)
    return await get_holdings_with_live_prices(user.id, db, finnhub, currency=currency)

@router.delete("/holdings/{holding_id}")
def delete_holding(holding_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    holding = db.query(Holding).filter_by(id=holding_id, user_id=user.id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    db.delete(holding)
    db.commit()
    return {"detail": "Deleted"}
