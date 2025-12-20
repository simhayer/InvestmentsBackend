from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from services.holding_service import get_all_holdings
from services.finnhub_news_service import get_company_news_for_symbols
from services.supabase_auth import get_current_db_user

router = APIRouter()

@router.get("/latest-for-user")
async def get_latest_news_for_user(user=Depends(get_current_db_user) ,db: Session = Depends(get_db), ):
    holdings = get_all_holdings(user.id, db)
    holdingSymbols = [str(h.symbol) for h in holdings if isinstance(h.symbol, str)]
    if not holdingSymbols:
        return {"news": []}
    
    return await get_company_news_for_symbols(holdingSymbols, days_back=7, limit_per_symbol=5)

@router.get("/latest-for-symbol")
async def get_latest_news_for_symbol(symbol: str, days_back: int = 7, limit_per_symbol: int = 5):
    return await get_company_news_for_symbols([symbol], days_back=days_back, limit_per_symbol=limit_per_symbol)