import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.user import User
from services.supabase_auth import get_current_db_user
from services.tier import get_user_plan
from database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CURRENCIES = {"USD", "CAD"} 

class MeOut(BaseModel):
  id: int
  supabase_user_id: str
  email: str
  base_currency: str  # or Literal["USD","CAD"]
  plan: str  # "free" | "premium" | "pro"

@router.get("/me", response_model=MeOut)
def me(
    current_user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    plan = get_user_plan(current_user, db)
    return MeOut(
        id=current_user.id,
        supabase_user_id=current_user.supabase_user_id,
        email=current_user.email,
        base_currency=current_user.currency or "USD",
        plan=plan,
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

    logger.info("currency_updated user_id=%s new_currency=%s", current_user.id, currency)
    return {
        "status": "success",
        "new_currency": currency,
        "base_currency_source": current_user.base_currency_source,
    }