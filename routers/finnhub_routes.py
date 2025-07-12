# finnhub_routes.py
from fastapi import APIRouter, Depends, HTTPException
import httpx, os
from auth import get_current_user  # reuse your auth
from dotenv import load_dotenv
import asyncio

load_dotenv()  # loads from .env if needed

router = APIRouter()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

@router.get("/price")
async def get_price(symbol: str, user=Depends(get_current_user)):
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            )
            response.raise_for_status()
            data = response.json()
            return {
                "symbol": symbol,
                "currentPrice": data.get("c"),
                "high": data.get("h"),
                "low": data.get("l"),
                "open": data.get("o"),
                "previousClose": data.get("pc")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price: {e}")


from fastapi import APIRouter, Depends, HTTPException, Query
import httpx, os
from auth import get_current_user
from typing import List
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
API_KEY = os.getenv("FINNHUB_API_KEY")

@router.post("/prices")
async def get_prices(symbols: List[str], user=Depends(get_current_user)):
    if not API_KEY or not symbols:
        raise HTTPException(status_code=400, detail="Missing symbols or API key")

    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}")
            for symbol in symbols
        ]
        responses = await asyncio.gather(*tasks)

    result = {}
    for symbol, res in zip(symbols, responses):
        try:
            data = res.json()
            result[symbol] = {
                "currentPrice": data.get("c"),
                "high": data.get("h"),
                "low": data.get("l"),
            }
        except Exception:
            result[symbol] = None

    return result

@router.get("/search")
async def search_symbols(query: str, user=Depends(get_current_user)):
    if not query:
        return []

    async with httpx.AsyncClient() as client:
        res = await client.get("https://finnhub.io/api/v1/search", params={
            "q": query,
            "token": API_KEY
        })
        data = res.json()
        return [
            item for item in data.get("result", [])
            if item.get("symbol") and item.get("description")
        ]

@router.get("/quote")
async def fetch_quote(symbol: str, user=Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        res = await client.get("https://finnhub.io/api/v1/quote", params={
            "symbol": symbol,
            "token": API_KEY
        })
        return res.json()

@router.get("/profile")
async def fetch_profile(symbol: str, user=Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        res = await client.get("https://finnhub.io/api/v1/stock/profile2", params={
            "symbol": symbol,
            "token": API_KEY
        })
        return res.json()