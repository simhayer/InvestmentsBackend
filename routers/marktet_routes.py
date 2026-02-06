# routes/market_overview.py
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session
from database import get_db
from services.market_service import etag_for, get_market_overview_cached, refresh_market_overview

router = APIRouter()

@router.get("/overview")
def get_market_overview(db: Session = Depends(get_db)):
    data = get_market_overview_cached(db, max_age_sec=60)
    return {"message": "Market overview data", "data": {"top_items": data["items"], "ai_summary": data.get("ai_summary")}}

# FIX - fix this when market pipeline is ready
# @router.get("/summary")
# def summary(request: Request, response: Response, db: Session = Depends(get_db)):
#     data, stored_at = get_market_summary_cached(db)  # stored_at = datetime (tz-aware)
#     etag = etag_for(data)

#     inm = request.headers.get("if-none-match")
#     if inm and inm == etag:
#         response.status_code = 304
#         return

#     # Caching headers
#     response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=1800"
#     response.headers["ETag"] = etag
#     response.headers["Last-Modified"] = stored_at.strftime("%a, %d %b %Y %H:%M:%S GMT")  # RFC 1123

#     return {
#         "message": "Market summary data",
#         "data": data,
#         "meta": {
#             "updated_at": stored_at.isoformat(),  # <- the one you’ll show as “Updated … ago”
#         },
#     }

# @router.post("/overview/refresh")
# def post_refresh_market_overview(db: Session = Depends(get_db)):
#     data = refresh_market_overview(db)
#     return {"ok": True, "items": data["items"]}
