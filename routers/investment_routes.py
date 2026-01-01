# routers/market_routes.py
from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool
from services.yahoo_service import (
    get_full_stock_data,
    get_price_history,
    get_financials,
    get_earnings,
    get_analyst,
    get_overview,
)

router = APIRouter()

@router.get("/quote/{symbol}")
async def get_quote(symbol: str, q: str | None = Query(default=None)):
    yahoo_symbol = (q or symbol or "").upper().strip()
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    data = await run_in_threadpool(get_full_stock_data, yahoo_symbol)
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON")
    if data.get("status") != "ok":
        msg = data.get("message") or "Failed to fetch quote data"
        code = 502 if data.get("error_code") == "YAHOOQUERY_FAILURE" else 400
        raise HTTPException(status_code=code, detail=msg)
    return data

@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    q: str | None = Query(default=None),
    period: str = Query("1y", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"),
):
    yahoo_symbol = (q or symbol or "").upper().strip()
    if not yahoo_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    data = await run_in_threadpool(get_price_history, yahoo_symbol, period, interval)
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", "Fetch failed"))
    return data

# NEW: Financial statements (annual/quarterly)
@router.get("/financials/{symbol}")
async def financials(
    symbol: str,
    period: str = Query("annual", pattern="^(annual|quarterly)$"),
):
    data = await run_in_threadpool(get_financials, symbol, period)
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", "Fetch failed"))
    return data

# NEW: Earnings & events
@router.get("/earnings/{symbol}")
async def earnings(symbol: str):
    data = await run_in_threadpool(get_earnings, symbol)
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", "Fetch failed"))
    return data

# NEW: Analyst targets & recommendation trend
@router.get("/analyst/{symbol}")
async def analyst(symbol: str):
    data = await run_in_threadpool(get_analyst, symbol)
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", "Fetch failed"))
    return data

# NEW: Company profile/overview
@router.get("/overview/{symbol}")
async def overview(symbol: str):
    data = await run_in_threadpool(get_overview, symbol)
    if data.get("status") != "ok":
        raise HTTPException(status_code=502, detail=data.get("message", "Fetch failed"))
    return data
