# finnhub_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from services.supabase_auth import get_current_db_user
from services.finnhub_service import (
    FinnhubService,
    FinnhubServiceError,
)

router = APIRouter()

# ---- Dependency to get the service (grabs API key from env inside service) ----
def get_finnhub_service() -> FinnhubService:
    return FinnhubService()


# ---------- Schemas ----------
class PriceRequest(BaseModel):
    symbols: List[str]
    types: List[str]
    currency: str = "USD"


# ---------- Routes (thin controllers delegating to the service) ----------
@router.get("/price")
async def get_price(
    symbol: str,
    type: str = "stock",
    user=Depends(get_current_db_user),
    svc: FinnhubService = Depends(get_finnhub_service),
):
    try:
        return await svc.get_price(symbol=symbol, typ=type)
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price: {e}")


@router.post("/prices")
async def get_prices(
    request: PriceRequest,
    user=Depends(get_current_db_user),
    svc: FinnhubService = Depends(get_finnhub_service),
):
    if len(request.symbols) != len(request.types):
        raise HTTPException(status_code=400, detail="Symbols and types list length mismatch")

    try:
        pairs = list(zip(request.symbols, request.types))
        return await svc.get_prices(pairs=pairs, currency=request.currency)
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching prices: {e}")


@router.get("/search")
async def search_symbols(
    query: str,
    svc: FinnhubService = Depends(get_finnhub_service),
):
    try:
        results = await svc.search_symbols(query)
        limit = 5
        return results[:limit]
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/quote")
async def fetch_quote(
    symbol: str,
    user=Depends(get_current_db_user),
    svc: FinnhubService = Depends(get_finnhub_service),
):
    try:
        return await svc.fetch_quote(symbol)
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profile")
async def fetch_profile(
    symbol: str,
    user=Depends(get_current_db_user),
    svc: FinnhubService = Depends(get_finnhub_service),
):
    try:
        return await svc.fetch_profile(symbol)
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))