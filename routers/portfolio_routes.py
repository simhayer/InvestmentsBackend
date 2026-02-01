# routers/portfolio_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from services.finnhub.finnhub_service import FinnhubService
from services.portfolio_service import get_portfolio_summary
from services.supabase_auth import get_current_db_user
from services.currency_service import resolve_currency
from services.portfolio.portfolio_health_score_service import build_portfolio_health_score
from services.portfolio.portfolio_health_explain_service import explain_portfolio_health
from schemas.portfolio_health_score import PortfolioHealthScoreResponse
from schemas.portfolio_health_explain import (
    PortfolioHealthExplainRequest,
    PortfolioHealthExplainResponse,
)
from typing import Literal
from models.user import User

router = APIRouter()

def get_finnhub_service() -> FinnhubService:
    return FinnhubService()

@router.get("/summary")
async def portfolio_summary(
    currency: str | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
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
    
@router.get("/health-score", response_model=PortfolioHealthScoreResponse)
async def portfolio_health_score_v2(
    currency: str = Query("USD", pattern="^(USD|CAD)$"),
    baseline: Literal["balanced", "growth", "conservative"] = Query("balanced"),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
    user: User = Depends(get_current_db_user),
):
    return await build_portfolio_health_score(
        user_id=str(user.id),
        db=db,
        finnhub=finnhub,
        currency=currency,
        baseline=baseline,
    )


@router.post("/health-explain", response_model=PortfolioHealthExplainResponse)
async def portfolio_health_explain(
    req: PortfolioHealthExplainRequest,
    user: User = Depends(get_current_db_user),
):
    return await explain_portfolio_health(req)
