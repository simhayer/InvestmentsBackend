# routers/holdings_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services.auth_service import get_current_user
import schemas.general as general, services.crud as crud
from models.holding import Holding

router = APIRouter()


@router.post("/holdings")
def save_holding(holding: general.HoldingCreate, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.create_holding(db, user.id, holding.symbol, holding.quantity, holding.purchase_price, holding.type)

@router.get("/holdings")
def get_holdings(db: Session = Depends(get_db), user=Depends(get_current_user)):
    print('Fetching holdings for user:', user.id)
    holdings = db.query(Holding).filter_by(user_id=user.id).all()

    print('Fetched holdings:', holdings)

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

@router.delete("/holdings/{holding_id}")
def delete_holding(holding_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    holding = db.query(Holding).filter_by(id=holding_id, user_id=user.id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    db.delete(holding)
    db.commit()
    return {"detail": "Deleted"}
