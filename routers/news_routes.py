from fastapi import APIRouter
from services.finnhub.finnhub_news_service import get_company_news_for_symbols

router = APIRouter()

@router.get("/latest-for-symbol")
async def get_latest_news_for_symbol(symbol: str, days_back: int = 7, limit_per_symbol: int = 5):
    # this is a function for an array, improve later if needed
    return await get_company_news_for_symbols([symbol], days_back=days_back, limit_per_symbol=limit_per_symbol)