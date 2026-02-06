from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from fastapi import APIRouter, Query
from models.user import User
from services.portfolio.portfolio_service import get_or_compute_portfolio_analysis
from services.supabase_auth import get_current_db_user
from services.finnhub.finnhub_service import FinnhubService
from routers.finnhub_routes import get_finnhub_service

router = APIRouter()

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    force: bool = Query(False, description="Bypass cache and recompute now"),
    user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    data, meta = await get_or_compute_portfolio_analysis(
        user_id=str(user.id),
        db=db,
        base_currency=user.currency,  # "USD" or "CAD"
        days_of_news=7,
        targets={"Equities": 60, "Bonds": 30, "Cash": 10},
        force=force,
        finnhub=finnhub,
    )
    if data is None:
        reason = meta.get("reason") if isinstance(meta, dict) else "unknown"
        raise HTTPException(status_code=400, detail=f"Cannot analyze portfolio: {reason}")

    return {
        "status": "ok",
        "user_id": user.id,
        **meta,
        # "ai_layers": unwrap_layers_for_ui(data["ai_layers"]),
    }

class SymbolReq(BaseModel):
    symbol: str