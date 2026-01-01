# finnhub_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from services.supabase_auth import get_current_db_user
from services.finnhub_service import (
    FinnhubService,
    FinnhubServiceError,
)
from cache.crypto_catalog import crypto_catalog

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
    q = (query or "").strip()
    if not q:
        return []

    limit = 5
    merged = []

    # 1) Crypto first (optional)
    crypto_hits = crypto_catalog.search(q, limit=limit)
    for c in crypto_hits:
        merged.append({
            "symbol": c.symbol,
            "description": f"{c.name} ({c.symbol}) • Crypto" if c.name else f"{c.symbol} • Crypto",
            "type": "crypto",
            "quote_symbol": f"{c.symbol}-USD",
        })

    # 2) Stocks from Finnhub
    try:
        stock_hits = await svc.search_symbols(q)
        for item in stock_hits:
            if isinstance(item, dict) and item.get("symbol") and item.get("description"):
                merged.append({
                    **item,
                    "asset_type": "stock",
                    "quote_symbol": item.get("symbol"),
                })
                if len(merged) >= limit * 3:  # allow room before dedupe
                    break
    except Exception:
        pass

    # 3) De-dupe by SYMBOL (so frontend key={symbol} is safe)
    seen_symbols = set()
    final = []
    for item in merged:
        sym = item.get("symbol")
        if not sym or sym in seen_symbols:
            continue
        seen_symbols.add(sym)
        final.append(item)
        if len(final) >= limit:
            break

    return final

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