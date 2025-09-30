from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from services.auth_service import get_current_user
from services.ai_service import analyze_investment_portfolio, analyze_investment_symbol_perplexity, analyze_portfolio_perplexity
from services.holding_service import get_all_holdings
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel

router = APIRouter()

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    holdings = get_all_holdings(user.id, db)
    return await run_in_threadpool(analyze_portfolio_perplexity, holdings)
    # return await run_in_threadpool(analyze_investment_portfolio, holdings)

class SymbolReq(BaseModel):
    symbol: str

@router.post("/analyze-symbol")
async def analyze_symbol_endpoint(req: SymbolReq, user=Depends(get_current_user)):
    return await run_in_threadpool(analyze_investment_symbol_perplexity, req.symbol)
