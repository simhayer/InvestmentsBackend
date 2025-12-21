# routers/portfolio_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from services.finnhub_service import FinnhubService
from services.portfolio_service import get_portfolio_summary
from services.supabase_auth import get_current_db_user
from services.currency_service import resolve_currency

router = APIRouter()

def get_finnhub_service() -> FinnhubService:
    return FinnhubService()


@router.get("/summary")
async def portfolio_summary(
    currency: str | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    resolved_currency = resolve_currency(user, currency)

    try:
        return await get_portfolio_summary(
            str(user.id),
            db,
            finnhub,
            currency=resolved_currency,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build portfolio summary: {e}")
