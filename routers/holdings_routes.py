# routers/holdings_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import models, schemas, crud
from models import Holding

router = APIRouter()

@router.post("/holdings")
def save_holding(holding: schemas.HoldingCreate, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return crud.create_holding(db, user.id, holding.symbol, holding.quantity, holding.avg_price, holding.type)

@router.get("/holdings")
def get_holdings(db: Session = Depends(get_db), user=Depends(get_current_user)):
    print('getting user for', user)
    return db.query(Holding).filter(Holding.user_id == user.id).all()

@router.delete("/holdings/{holding_id}")
def delete_holding(holding_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    holding = db.query(models.Holding).filter_by(id=holding_id, user_id=user.id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    db.delete(holding)
    db.commit()
    return {"detail": "Deleted"}
