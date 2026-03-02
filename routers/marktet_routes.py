# routes/market_overview.py
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session
from database import get_db
from middleware.rate_limit import limiter
from services.market_service import etag_for, get_market_overview_cached, refresh_market_overview
from services.global_brief_service import get_global_brief_cached, get_predictions_cached
from services.market_monitor_service import get_market_monitor_panel, get_personalized_market_monitor_panel
from services.currency_service import resolve_currency
from services.supabase_auth import get_current_db_user
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService
from models.user import User
from schemas.market_monitor import MarketMonitorEnvelope, PersonalizedMarketMonitorEnvelope

router = APIRouter()


@router.get("/overview")
def get_market_overview(db: Session = Depends(get_db)):
    data = get_market_overview_cached(db, max_age_sec=60)
    return {"message": "Market overview data", "data": {"top_items": data["items"], "ai_summary": data.get("ai_summary")}}


@router.get("/global-brief")
@limiter.limit("20/minute")
async def get_global_brief(request: Request, refresh: bool = False):
    """Public endpoint: AI-generated global market brief for Finance World page. Cached 30 min. ?refresh=true bypasses cache."""
    data = await get_global_brief_cached(force_refresh=refresh)
    return {"message": "Global market brief", "data": data, "meta": {"updated_at": data.get("as_of")}}


@router.get("/predictions")
@limiter.limit("20/minute")
async def get_predictions(request: Request, refresh: bool = False):
    """Public endpoint: short-term market outlook/sentiment. Cached 30 min. ?refresh=true bypasses cache (and refreshes brief)."""
    data = await get_predictions_cached(force_refresh=refresh)
    return {"message": "Market predictions", "data": data}


@router.get("/monitor-panel", response_model=MarketMonitorEnvelope)
@limiter.limit("20/minute")
async def get_monitor_panel(request: Request, refresh: bool = False, db: Session = Depends(get_db)):
    """Public endpoint: composed payload for a finance-monitor-style left panel."""
    data = await get_market_monitor_panel(db, force_refresh=refresh)
    return {"message": "Market monitor panel", "data": data}


@router.get("/monitor-panel/personalized", response_model=PersonalizedMarketMonitorEnvelope)
@limiter.limit("20/minute")
async def get_personalized_monitor_panel(
    request: Request,
    refresh: bool = False,
    currency: str | None = None,
    watchlist_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """Authenticated endpoint: global monitor plus portfolio-aware or symbol-override personalization."""
    resolved_currency = resolve_currency(user, currency)
    try:
        data = await get_personalized_market_monitor_panel(
            db,
            user_id=str(user.id),
            finnhub=finnhub,
            currency=resolved_currency,
            force_refresh=refresh,
            watchlist_id=watchlist_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": "Personalized market monitor panel", "data": data}


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
