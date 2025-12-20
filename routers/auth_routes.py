from fastapi import APIRouter, Depends
from services.supabase_auth import get_current_db_user
router = APIRouter()

@router.get("/me")
async def me(payload: dict = Depends(get_current_db_user)):
    return {
        "id": payload.get("sub"),      # Supabase user id (UUID string)
        "email": payload.get("email"),
    }
