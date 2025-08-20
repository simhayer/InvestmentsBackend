# finnhub_routes.py
from fastapi import APIRouter, Depends, HTTPException
import httpx, os
from services.auth_service import get_current_user
from dotenv import load_dotenv
import asyncio
from fastapi import APIRouter, Depends, HTTPException
import httpx, os
from services.auth_service import get_current_user
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def format_finnhub_symbol(symbol: str, is_crypto: bool = False) -> str:
    if is_crypto:
        return f"BINANCE:{symbol.upper()}USDT"
    return symbol.upper()

async def get_usd_to_cad_rate(client: httpx.AsyncClient) -> float:
    try:
        res = await client.get("https://api.frankfurter.app/latest?from=USD&to=CAD")
        data = res.json()

        rate = data.get("rates", {}).get("CAD")
        if rate is None:
            raise ValueError("CAD rate missing")

        return rate

    except Exception as e:
        print("⚠️ Failed to fetch USD to CAD rate:", e)
        return 1.0

    
@router.get("/price")
async def get_price(symbol: str, type: str = "stock", user=Depends(get_current_user)):
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    formatted_symbol = format_finnhub_symbol(symbol, is_crypto=(type == "cryptocurrency"))

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://finnhub.io/api/v1/quote?symbol={formatted_symbol}&token={FINNHUB_API_KEY}"
            )
            response.raise_for_status()
            data = response.json()

            current_price = data.get("c")
            if current_price is None or current_price == 0:
                raise HTTPException(status_code=404, detail="Price not available for this symbol")

            return {
                "symbol": symbol,
                "formattedSymbol": formatted_symbol,
                "currentPrice": current_price,
                "high": data.get("h"),
                "low": data.get("l"),
                "open": data.get("o"),
                "previousClose": data.get("pc")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")


class PriceRequest(BaseModel):
    symbols: List[str]
    types: List[str]
    currency: str = "USD"

@router.post("/prices")
async def get_prices(request: PriceRequest, user=Depends(get_current_user)):
    symbols = request.symbols
    types = request.types
    currency = request.currency

    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=400, detail="Missing API key")

    if len(symbols) != len(types):
        raise HTTPException(status_code=400, detail="Symbols and types list length mismatch")

    clean = [(s.strip(), t) for s, t in zip(symbols, types) if s and s.strip()]
    if not clean:
        raise HTTPException(status_code=400, detail="No valid symbols provided")

    formatted_symbols = [format_finnhub_symbol(s, t == "cryptocurrency") for s, t in clean]

    async with httpx.AsyncClient() as client:
        usd_to_cad = await get_usd_to_cad_rate(client) if currency.upper() == "CAD" else 1.0

        tasks = [
            client.get(f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_API_KEY}")
            for sym in formatted_symbols
        ]
        responses = await asyncio.gather(*tasks)

    result = {}
    for (original_symbol, _), formatted_symbol, res in zip(clean, formatted_symbols, responses):
        try:
            data = res.json()
            usd_price = data.get("c")
            if usd_price and usd_price != 0:
                final_price = round(usd_price * usd_to_cad, 2)
                result[original_symbol] = {
                    "currentPrice": final_price,
                    "currency": currency.upper(),
                    "high": round(data.get("h", 0) * usd_to_cad, 2),
                    "low": round(data.get("l", 0) * usd_to_cad, 2),
                    "formattedSymbol": formatted_symbol
                }
            else:
                result[original_symbol] = None
        except Exception:
            result[original_symbol] = None

    return result

@router.get("/search")
async def search_symbols(query: str, user=Depends(get_current_user)):
    if not query:
        return []

    async with httpx.AsyncClient() as client:
        res = await client.get("https://finnhub.io/api/v1/search", params={
            "q": query,
            "token": FINNHUB_API_KEY
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
            "token": FINNHUB_API_KEY
        })
        return res.json()

@router.get("/profile")
async def fetch_profile(symbol: str, user=Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        res = await client.get("https://finnhub.io/api/v1/stock/profile2", params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        })
        return res.json()
    
async def fetch_prices_for_symbols(symbols: list[str]) -> dict[str, float]:
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Finnhub API key")

    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}")
            for symbol in symbols
        ]
        responses = await asyncio.gather(*tasks)

    prices = {}
    for symbol, res in zip(symbols, responses):
        try:
            data = res.json()
            prices[symbol] = data.get("c", 0)
        except Exception:
            prices[symbol] = 0

    return prices