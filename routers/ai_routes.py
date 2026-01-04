from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, Optional
from services.linkup.agents.single_stock_analysis_agent import analyze_stock_async
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from fastapi import APIRouter, Query
from models.user import User
from models.holding import to_dto
from services.holding_service import get_all_holdings
from services.portfolio_service import get_or_compute_portfolio_analysis
from services.supabase_auth import get_current_db_user
from services.finnhub_service import FinnhubService
from routers.finnhub_routes import get_finnhub_service
from utils.common_helpers import unwrap_layers_for_ui

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
        "ai_layers": unwrap_layers_for_ui(data["ai_layers"]),
    }

class SymbolReq(BaseModel):
    symbol: str
    base_currency: Optional[str] = None
    metrics_for_symbol: Optional[Dict[str, Any]] = None
    user_request: Optional[str] = None
    needs_filings: Optional[bool] = False
    cik: Optional[str] = None

@router.post("/analyze-symbol")
async def analyze_symbol_endpoint(
    req: SymbolReq,
    user=Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    # return dummy_holding_response
    base_currency = req.base_currency or getattr(user, "currency", None) or "USD"
    holdings_rows = get_all_holdings(str(user.id), db)
    holdings = [to_dto(h) for h in holdings_rows] if holdings_rows else []
    return await analyze_stock_async(
        req.symbol,
        base_currency,
        req.metrics_for_symbol,
        holdings,
        user_request=req.user_request,
        needs_filings=bool(req.needs_filings),
        cik=req.cik,
    )
