# routes/market_overview.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from services.market_service import get_market_overview_cached, refresh_market_overview

router = APIRouter()

@router.get("/overview")
def get_market_overview(db: Session = Depends(get_db)):
    data = get_market_overview_cached(db, max_age_sec=60)
    return {"message": "Market overview data", "data": {"top_items": data["items"], "ai_summary": data.get("ai_summary")}}

@router.post("/overview/refresh")
def post_refresh_market_overview(db: Session = Depends(get_db)):
    data = refresh_market_overview(db)
    return {"ok": True, "items": data["items"]}
