from fastapi import APIRouter, Request
from middleware.rate_limit import limiter
from services.finnhub.finnhub_news_service import (
    get_company_news_for_symbols,
    get_global_news_cached,
)

router = APIRouter()


@router.get("/latest-for-symbol")
async def get_latest_news_for_symbol(symbol: str, days_back: int = 7, limit_per_symbol: int = 5):
    # this is a function for an array, improve later if needed
    return await get_company_news_for_symbols([symbol], days_back=days_back, limit_per_symbol=limit_per_symbol)


@router.get("/global")
@limiter.limit("30/minute")
async def get_global_news(request: Request, category: str = "general", limit: int = 20):
    """Public endpoint: general market news for Finance World page. Category: general, crypto, forex, merger, etc."""
    if limit > 50:
        limit = 50
    items = await get_global_news_cached(category=category, limit=limit)
    return {"items": items}