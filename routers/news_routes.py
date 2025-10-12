from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from services.auth_service import get_current_user
from services.linkup_service import fetch_latest_news_for_holdings
from services.holding_service import get_all_holdings
from services.finnhub_news_service import get_company_news_for_symbols
from typing import List

router = APIRouter()

@router.get("/latest-for-user")
async def get_latest_news_for_user(user=Depends(get_current_user) ,db: Session = Depends(get_db), ):
    holdings = get_all_holdings(user.id, db)
    holdingSymbols = [str(h.get("symbol")) for h in holdings if isinstance(h.get("symbol"), str)]
    if not holdingSymbols:
        return {"news": []}
    
    return await get_company_news_for_symbols(holdingSymbols, days_back=7, limit_per_symbol=5)