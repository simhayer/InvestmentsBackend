from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from services.auth_service import get_current_user
from services.ai_service import analyze_investment_portfolio, analyze_investment_symbol_perplexity, analyze_portfolio_perplexity
from services.holding_service import get_all_holdings
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from fastapi import APIRouter, Body, Query
from services.portfolio_summary import summarize_portfolio_news
from services.finnhub_news_service import get_company_news_for_symbols
from typing import List

router = APIRouter()

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    holdings = get_all_holdings(user.id, db)
    return await run_in_threadpool(analyze_portfolio_perplexity, holdings)

class SymbolReq(BaseModel):
    symbol: str

@router.post("/analyze-symbol")
async def analyze_symbol_endpoint(req: SymbolReq, user=Depends(get_current_user)):
    return await run_in_threadpool(analyze_investment_symbol_perplexity, req.symbol)

@router.post("/news-summary")
async def portfolio_news_summary(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    days_back: int = Query(7, ge=1, le=30),
    per_symbol_limit: int = Query(6, ge=1, le=20),
):
    holdings = get_all_holdings(user.id, db)
    symbols = [h.symbol for h in holdings]
    if not symbols:
        return {"summary": "", "highlights": [], "risks": [], "per_symbol": {}, "sentiment": 0, "sources": []}

    # 1) Fetch recent news per symbol
    news_by_symbol = await get_company_news_for_symbols(
        symbols, days_back=days_back, limit_per_symbol=per_symbol_limit
    )

    # 2) Summarize as a portfolio brief
    news_by_symbol_dict = {
        symbol: [item.__dict__ if hasattr(item, "__dict__") else dict(item) for item in items]
        for symbol, items in news_by_symbol.items()
    }
    summary_text = await summarize_portfolio_news(news_by_symbol_dict, symbols=symbols)

    urls: list[str] = []
    for arr in news_by_symbol.values():
        for it in arr:
            u = it.get("url")
            if u and u not in urls:
                urls.append(u)

    return {
        "summary": summary_text,
        "highlights": [],       # (optional) you can derive separately later
        "risks": [],            # (optional)
        "per_symbol": {},       # (optional)
        "sentiment": 0,         # (optional)
        "sources": urls[:8],
    }