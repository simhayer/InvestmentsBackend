# routers/v2/analyze_portfolio_routes.py
import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService
from services.supabase_auth import get_current_db_user

from services.cache.cache_backend import cache_get, cache_set
from services.ai.portfolio.portfolio_analysis import (
    run_portfolio_analysis_task,
    TTL_PORTFOLIO_TASK_RESULT_SEC,
)

router = APIRouter()


def _ck_task(task_id: str) -> str:
    return f"PORTFOLIO:ANALYZE:TASK:{(task_id or '').strip()}"


# Optional: portfolio report cache key (if you also cache the latest report)
def _ck_portfolio_report(user_id: int | str, currency: str) -> str:
    return f"PORTFOLIO:REPORT:{user_id}:{(currency or 'USD').strip().upper()}"


# ----------------------------
# Routes
# ----------------------------
@router.post("/analyze-portfolio")
async def start_portfolio_analysis(
    bg: BackgroundTasks,
    currency: str = Query("USD", description="Portfolio currency view (USD/CAD)"),
    force: bool = Query(False, description="Bypass cached report and recompute now"),
    user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    curr = (currency or "USD").strip().upper()
    if curr not in ("USD", "CAD"):
        raise HTTPException(status_code=400, detail="Invalid currency (USD/CAD only)")

    # If you want: quick return cached report when not forcing
    if not force:
        cached = cache_get(_ck_portfolio_report(user.id, curr))
        if cached:
            return {
                "task_id": None,
                "status": "complete",
                "data": {"report": cached, "cached": True},
            }

    task_id = f"task_portfolio_{user.id}_{curr}_{os.urandom(4).hex()}"
    task_key = _ck_task(task_id)

    cache_set(
        task_key,
        {"status": "processing", "data": None},
        ttl_seconds=TTL_PORTFOLIO_TASK_RESULT_SEC,
    )

    # Background task runs async function (Starlette supports this)
    bg.add_task(
        run_portfolio_analysis_task,
        user_id=str(user.id),
        task_id=task_id,
        currency=curr,
        force=force,
        # Pass through deps so the task doesn't need to recreate them
        db=db,
        finnhub=finnhub,
    )

    return {"task_id": task_id, "status": "started"}


@router.get("/portfolio-status/{task_id}")
async def get_portfolio_status(
    task_id: str,
    debug: int = Query(0),
):
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task_id")

    task_key = _ck_task(task_id)
    result = cache_get(task_key)

    if not result:
        raise HTTPException(status_code=404, detail="Task not found (expired or invalid)")

    if isinstance(result, dict):
        if debug:
            return result
        return {"status": result.get("status"), "data": result.get("data")}

    return {"status": "failed", "data": {"error": "Invalid task payload"}}
