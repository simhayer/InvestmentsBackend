from fastapi import APIRouter, Depends
from pydantic import BaseModel
from models.user import User
from services.supabase_auth import get_current_db_user
from database import get_db
from sqlalchemy.orm import Session
from fastapi import HTTPException
router = APIRouter()

ALLOWED_CURRENCIES = {"USD", "CAD"} 

class MeOut(BaseModel):
  id: int
  supabase_user_id: str
  email: str
  base_currency: str  # or Literal["USD","CAD"]

@router.get("/me", response_model=MeOut)
def me(current_user: User = Depends(get_current_db_user)):
    return MeOut(
        id=current_user.id,
        supabase_user_id=current_user.supabase_user_id,
        email=current_user.email,
        base_currency=current_user.currency or "USD",
    )


class CurrencyUpdate(BaseModel):
    new_currency: str
    
@router.post("/update_currency")
def update_currency(payload: CurrencyUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_db_user)):
    currency = payload.new_currency.strip().upper()

    if currency not in ALLOWED_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported currency: {currency}",
        )

    current_user.currency = currency
    current_user.base_currency_source = "manual"

    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return {
        "status": "success",
        "new_currency": currency,
        "base_currency_source": current_user.base_currency_source,
    }