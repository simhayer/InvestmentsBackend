from fastapi import APIRouter, Depends
from pydantic import BaseModel
from models.user import User
from services.supabase_auth import get_current_db_user

router = APIRouter()

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
        base_currency=current_user.currency,
    )
