# services/finnhub_service.py
from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv

load_dotenv()


class FinnhubServiceError(Exception):
    """Domain-level error for the Finnhub service."""


def format_finnhub_symbol(symbol: str, is_crypto: bool = False) -> str:
    """
    Finnhub expects crypto as 'EXCHANGE:PAIR', e.g. BINANCE:BTCUSDT.
    """
    if is_crypto:
        return f"BINANCE:{symbol.upper()}USDT"
    return symbol.upper()


class FinnhubService:
    """
    A small, framework-agnostic async service to interact with Finnhub and related endpoints.
    You can reuse this from your routers, background tasks, or other services.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        fx_ttl_sec: int = 60,   # cache USD->CAD briefly (optional)
    ):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise FinnhubServiceError("Missing FINNHUB_API_KEY")

        self.timeout = timeout
        self._fx_ttl = fx_ttl_sec
        self._fx_cache: Optional[Tuple[float, float]] = None  # (rate, expires_at)

    @asynccontextmanager
    async def _client(self, client: Optional[httpx.AsyncClient] = None):
        if client is not None:
            yield client
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                yield c

    # ---------- FX ----------
    async def get_usd_to_cad_rate(self, client: Optional[httpx.AsyncClient] = None) -> float:
        now = time.time()
        if self._fx_cache and self._fx_cache[1] > now:
            return self._fx_cache[0]

        async with self._client(client) as c:
            try:
                r = await c.get("https://api.frankfurter.app/latest?from=USD&to=CAD")
                data = r.json()
                rate = data.get("rates", {}).get("CAD")
                if rate is None:
                    raise FinnhubServiceError("CAD rate missing from FX response")
                rate = float(rate)
                if self._fx_ttl:
                    self._fx_cache = (rate, now + self._fx_ttl)
                return rate
            except Exception as e:
                # If FX fails, fall back to 1.0 rather than exploding the whole request.
                # You can choose to re-raise if you want hard failures.
                print("⚠️ FX fetch failed, defaulting to 1.0:", e)
                return 1.0

    # ---------- Single price ----------
    async def get_price(
        self,
        symbol: str,
        typ: str = "stock",
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        formatted_symbol = format_finnhub_symbol(symbol, is_crypto=(typ == "cryptocurrency"))
        async with self._client(client) as c:
            r = await c.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": formatted_symbol, "token": self.api_key},
            )
            r.raise_for_status()
            data = r.json()
            current_price = data.get("c")
            if current_price is None or current_price == 0:
                raise FinnhubServiceError("Price not available for this symbol")

            return {
                "symbol": symbol,
                "formattedSymbol": formatted_symbol,
                "currentPrice": current_price,
                "high": data.get("h"),
                "low": data.get("l"),
                "open": data.get("o"),
                "previousClose": data.get("pc"),
            }

    # ---------- Batch prices with optional CAD conversion ----------
    async def get_prices(
        self,
        pairs: Iterable[Tuple[str, str]],  # (symbol, type)
        currency: str = "USD",
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        clean: List[Tuple[str, str]] = [(s.strip(), t) for s, t in pairs if s and s.strip()]
        if not clean:
            return {}

        formatted_symbols = [format_finnhub_symbol(s, t == "cryptocurrency") for s, t in clean]

        async with self._client(client) as c:
            usd_to_cad = await self.get_usd_to_cad_rate(c) if currency.upper() == "CAD" else 1.0
            tasks = [
                c.get("https://finnhub.io/api/v1/quote", params={"symbol": fs, "token": self.api_key})
                for fs in formatted_symbols
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        result: Dict[str, Optional[Dict[str, Any]]] = {}
        for (original_symbol, _), formatted_symbol, res in zip(clean, formatted_symbols, responses):
            try:
                if isinstance(res, Exception):
                    raise res

                assert isinstance(res, httpx.Response)
                data = res.json()
                usd_price = data.get("c"); pc = data.get("pc"); h = data.get("h"); l = data.get("l")
                if usd_price and usd_price != 0:
                    if currency.upper() == "CAD":
                        result[original_symbol] = {
                            "currentPrice": round(usd_price * usd_to_cad, 2),
                            "currency": "CAD",
                            "previousClose": round((pc or 0) * usd_to_cad, 2),
                            "high": round((h or 0) * usd_to_cad, 2),
                            "low": round((l or 0) * usd_to_cad, 2),
                            "formattedSymbol": formatted_symbol,
                        }
                    else:
                        # keep USD values as-is (matching your current behavior)
                        result[original_symbol] = {
                            "currentPrice": usd_price,
                            "currency": "USD",
                            "previousClose": pc,
                            "high": h,
                            "low": l,
                            "formattedSymbol": formatted_symbol,
                        }
                else:
                    result[original_symbol] = None
            except Exception:
                result[original_symbol] = None

        return result

    # ---------- Utility wrappers ----------
    async def search_symbols(self, query: str, client: Optional[httpx.AsyncClient] = None) -> List[Dict[str, Any]]:
        if not query:
            return []
        async with self._client(client) as c:
            try:
                r = await c.get(
                    "https://finnhub.io/api/v1/search",
                    params={"q": query, "token": self.api_key},
                )
                if r.status_code >= 400:
                    return []
                try:
                    data = r.json()
                except ValueError:
                    return []
                return [item for item in data.get("result", []) if item.get("symbol") and item.get("description")]
            except Exception:
                return []

    async def fetch_quote(self, symbol: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        async with self._client(client) as c:
            r = await c.get("https://finnhub.io/api/v1/quote", params={"symbol": symbol, "token": self.api_key})
            return r.json()

    async def fetch_profile(self, symbol: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        async with self._client(client) as c:
            r = await c.get("https://finnhub.io/api/v1/stock/profile2", params={"symbol": symbol, "token": self.api_key})
            return r.json()

    async def fetch_prices_for_symbols(
        self, symbols: List[str], client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, float]:
        if not symbols:
            return {}

        async with self._client(client) as c:
            tasks = [
                c.get("https://finnhub.io/api/v1/quote", params={"symbol": s, "token": self.api_key})
                for s in symbols
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        prices: Dict[str, float] = {}
        for symbol, res in zip(symbols, responses):
            try:
                if isinstance(res, Exception):
                    raise res

                assert isinstance(res, httpx.Response)
                data = res.json()
                prices[symbol] = float(data.get("c") or 0.0)
            except Exception:
                prices[symbol] = 0.0
        return prices
