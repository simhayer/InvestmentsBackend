# routers/v2/analyse_symbol_routes.py
import os
from fastapi import BackgroundTasks, HTTPException, APIRouter, Query
from services.cache.cache_backend import cache_get, cache_set
from services.ai.analyze_symbol.analyze_symbol_service import run_analysis_task, TTL_TASK_RESULT_SEC, _ck_task
router = APIRouter()

# ----------------------------
# 10) Routes
# ----------------------------
@router.post("/analyze/{symbol}")
async def start_analysis(symbol: str, bg: BackgroundTasks):
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise HTTPException(status_code=400, detail="Missing symbol")

    task_id = f"task_{clean_symbol}_{os.urandom(4).hex()}"
    task_key = _ck_task(task_id)

    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_TASK_RESULT_SEC)

    # FastAPI/Starlette can run async background tasks; this is OK.
    bg.add_task(run_analysis_task, clean_symbol, task_id)

    return {"task_id": task_id, "status": "started"}

@router.get("/status/{task_id}")
async def get_status(task_id: str, debug: int = Query(0)):
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task_id")

    task_key = _ck_task(task_id)
    result = cache_get(task_key)

    if not result:
        raise HTTPException(status_code=404, detail="Task not found (expired or invalid)")

    if isinstance(result, dict):
        # expected shape: {"status": "...", "data": ...}
        if debug:
            return result
        return {"status": result.get("status"), "data": result.get("data")}

    # fallback if something weird got stored
    return {"status": "failed", "data": {"error": "Invalid task payload"}}