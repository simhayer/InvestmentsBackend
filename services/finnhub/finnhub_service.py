# services/finnhub_service.py
from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from utils.common_helpers import canonical_key, safe_json, normalize_asset_type
from services.cache.cache_backend import cache_get_many, cache_set_many
import finnhub

TTL_FINNHUB_QUOTE_SEC = 60

def _ck_quote(formatted_symbol: str) -> str:
    return f"FINNHUB:QUOTE:{(formatted_symbol or '').strip().upper()}"

import httpx
from dotenv import load_dotenv
load_dotenv()

class FinnhubServiceError(Exception):
    """Domain-level error for the Finnhub service."""

def format_finnhub_symbol(symbol: str, typ: str = "") -> str:
    """
    Finnhub quote endpoint expects:
      - Stocks/ETFs: "AAPL"
      - Crypto often as "EXCHANGE:PAIR" (e.g., "BINANCE:BTCUSDT")
    """
    s = (symbol or "").upper().strip()
    t = (typ or "").lower().strip()
    if not s:
        return s
    if t == "cryptocurrency":
        # If the symbol already looks qualified (e.g., "BINANCE:BTCUSDT"), keep it.
        if ":" in s:
            return s
        return f"BINANCE:{s}USDT"
    return s

@dataclass(frozen=True)
class Quote:
    current_price: float
    currency: str
    previous_close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    formatted_symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currentPrice": self.current_price,
            "currency": self.currency,
            "previousClose": self.previous_close,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "formattedSymbol": self.formatted_symbol,
        }

class FinnhubService:
    """
    Framework-agnostic async service for Finnhub quotes/search/profile.

    Design goals:
      - Always return quote maps keyed by canonical_key(symbol, type)
      - Concurrency limiting to reduce 429/rate-limit risk
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        max_concurrency: int = 8,
    ):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise FinnhubServiceError("Missing FINNHUB_API_KEY")

        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))

    @asynccontextmanager
    async def _client(self, client: Optional[httpx.AsyncClient] = None):
        if client is not None:
            yield client
            return
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            yield c

    def _auth_params(self, **params: Any) -> Dict[str, Any]:
        return {**params, "token": self.api_key}
    
    def get_finnhub_client(self) -> finnhub.Client:
        """Returns a synchronous Finnhub client using the stored API key."""
        if not self.api_key:
            raise FinnhubServiceError("Missing FINNHUB_API_KEY")
        return finnhub.Client(api_key=self.api_key)

    # -----------------------
    # Quote (single)
    # -----------------------

    async def get_price(
        self,
        symbol: str,
        typ: str = "stock",
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Returns a normalized quote dict for ONE symbol.
        Raises FinnhubServiceError when price is unavailable or request fails.
        """
        sym = (symbol or "").strip()
        if not sym:
            raise FinnhubServiceError("Missing symbol")

        formatted_symbol = format_finnhub_symbol(sym, typ)

        async with self._client(client) as c:
            r = await c.get(
                f"{self.BASE_URL}/quote",
                params=self._auth_params(symbol=formatted_symbol),
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise FinnhubServiceError(f"Finnhub quote failed: {e.response.status_code}") from e

            data = safe_json(r) or {}
            current_price = data.get("c")

            if current_price in (None, 0):
                raise FinnhubServiceError("Price not available for this symbol")

            q = Quote(
                current_price=float(current_price),
                currency="USD",
                previous_close=float(data["pc"]) if data.get("pc") not in (None,) else None,
                high=float(data["h"]) if data.get("h") not in (None,) else None,
                low=float(data["l"]) if data.get("l") not in (None,) else None,
                open=float(data["o"]) if data.get("o") not in (None,) else None,
                formatted_symbol=formatted_symbol,
            )
            payload = {"symbol": sym, "formattedSymbol": formatted_symbol, **q.to_dict()}
            return payload

    # -----------------------
    # Quote (batch)
    # -----------------------

    async def get_prices(
        self,
        pairs: Iterable[Tuple[str, str]],  # (symbol, type)
        currency: str = "USD",
        client: Optional[httpx.AsyncClient] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Batch quote fetch.
        Returns:
          {
            "AAPL:equity": {"currentPrice":..., "currency":"USD", ...} | None,
            "BTC:cryptocurrency": {...} | None
          }
        """
        cur = (currency or "USD").upper()

        clean: List[Tuple[str, str]] = []
        for s, t in pairs:
            sym = (s or "").strip()
            if not sym:
                continue
            clean.append((sym, normalize_asset_type(t) or "stock"))

        if not clean:
            return {}

        keys = [canonical_key(sym, typ) for sym, typ in clean]
        formatted = [format_finnhub_symbol(sym, typ) for sym, typ in clean]

        sem = asyncio.Semaphore(max(1, int(max_concurrency or self.max_concurrency)))

        async def fetch_one(c: httpx.AsyncClient, fs: str) -> Optional[Quote]:
            async with sem:
                r = await c.get(f"{self.BASE_URL}/quote", params=self._auth_params(symbol=fs))
                r.raise_for_status()
                data = safe_json(r)
                if not data:
                    return None

                cpx = data.get("c")
                if cpx in (None, 0):
                    return None

                # Finnhub quote endpoint is effectively USD for US stocks; crypto is in USDT.
                # We report USD (or USDT-like) as "USD" to match your current pipeline expectations.
                # If you want strictness later, return "USDT" for crypto and convert elsewhere.
                return Quote(
                    current_price=float(cpx),
                    currency="USD",
                    previous_close=float(data["pc"]) if data.get("pc") not in (None,) else None,
                    high=float(data["h"]) if data.get("h") not in (None,) else None,
                    low=float(data["l"]) if data.get("l") not in (None,) else None,
                    open=float(data["o"]) if data.get("o") not in (None,) else None,
                    formatted_symbol=fs,
                )

        async with self._client(client) as c:
            tasks = [fetch_one(c, fs) for fs in formatted]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out: Dict[str, Optional[Dict[str, Any]]] = {}
        for k, fs, res in zip(keys, formatted, results):
            if isinstance(res, Exception):
                out[k] = None
                continue
            if res is None or not isinstance(res, Quote):
                out[k] = None
                continue

            d = res.to_dict()
            # If the caller requested a currency other than USD, conversion belongs elsewhere.
            # We still echo requested currency via the caller layer if needed.
            d["currency"] = "USD"  # keep stable for now
            d["formattedSymbol"] = fs
            out[k] = d

        # If you want to enforce only USD requests here (recommended), you can:
        # if cur != "USD": raise FinnhubServiceError("Currency conversion must be done outside FinnhubService")

        return out
    
    async def get_prices_cached(
    self,
    pairs: Iterable[Tuple[str, str]],  # (symbol, type)
    currency: str = "USD",
    client: Optional[httpx.AsyncClient] = None,
    *,
    max_concurrency: Optional[int] = None,
    ttl_seconds: int = TTL_FINNHUB_QUOTE_SEC,
    ) -> Dict[str, Optional[Dict[str, Any]]]:

        def normalize_asset_type(typ: str | None) -> str:
            t = (typ or "").strip().lower()
            if not t:
                return "equity"
            if t in {"stock", "equity", "etf", "common stock", "adr"}:
                return "equity"
            if t in {"crypto", "cryptocurrency"}:
                return "cryptocurrency"
            return t

        # 1) normalize input, and precompute EVERYTHING once
        items: List[Dict[str, str]] = []
        for s, t in pairs:
            sym = (s or "").strip()
            if not sym:
                continue

            typ_n = normalize_asset_type(t)
            fs = format_finnhub_symbol(sym, typ_n)

            items.append({
                "sym": sym,
                "typ": typ_n,  # ✅ normalized
                "fs": fs,
                "out_key": canonical_key(sym, typ_n),  # ✅ matches get_prices()
                "cache_key": _ck_quote(fs),            # ok to key by formatted symbol
            })

        if not items:
            return {}

        # 2) bulk cache read (by cache_key)
        cache_keys = [it["cache_key"] for it in items]
        cached_map = cache_get_many(cache_keys)

        out: Dict[str, Optional[Dict[str, Any]]] = {}
        misses: List[Tuple[str, str]] = []
        miss_items: List[Dict[str, str]] = []

        for it in items:
            hit = cached_map.get(it["cache_key"])
            if isinstance(hit, dict):
                out[it["out_key"]] = hit
            else:
                out[it["out_key"]] = None
                misses.append((it["sym"], it["typ"]))  # ✅ normalized type
                miss_items.append(it)

        if not misses:
            return out

        # 3) fetch only misses from Finnhub (returns dict keyed by canonical_key(sym, typ))
        fresh = await self.get_prices(
            pairs=misses,
            currency=currency,
            client=client,
            max_concurrency=max_concurrency,
        )

        # 4) merge fresh into out + write-through cache per formattedSymbol
        write_back: Dict[str, Any] = {}
        for it in miss_items:
            payload = fresh.get(it["out_key"])
            out[it["out_key"]] = payload

            # optional: don’t cache None (avoid sticky missing)
            if isinstance(payload, dict):
                write_back[it["cache_key"]] = payload

        if write_back:
            cache_set_many(write_back, ttl_seconds=ttl_seconds)

        return out

    async def search_symbols(
        self,
        query: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        async with self._client(client) as c:
            try:
                r = await c.get(f"{self.BASE_URL}/search", params=self._auth_params(q=q))
                r.raise_for_status()
                data = safe_json(r) or {}
                results = data.get("result", [])
                if not isinstance(results, list):
                    return []
                return [
                    item
                    for item in results
                    if isinstance(item, dict) and item.get("symbol") and item.get("description")
                ]
            except Exception:
                return []

    async def fetch_quote(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        sym = (symbol or "").strip()
        if not sym:
            return {}
        async with self._client(client) as c:
            try:
                r = await c.get(f"{self.BASE_URL}/quote", params=self._auth_params(symbol=sym))
                r.raise_for_status()
                return safe_json(r) or {}
            except Exception:
                return {}

    async def fetch_profile(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        sym = (symbol or "").strip()
        if not sym:
            return {}
        async with self._client(client) as c:
            try:
                r = await c.get(f"{self.BASE_URL}/stock/profile2", params=self._auth_params(symbol=sym))
                r.raise_for_status()
                return safe_json(r) or {}
            except Exception:
                return {}

    async def fetch_basic_financials(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        sym = (symbol or "").strip()
        if not sym:
            return {}
        async with self._client(client) as c:
            try:
                r = await c.get(
                    f"{self.BASE_URL}/stock/metric",
                    params=self._auth_params(symbol=sym, metric="all"),
                )
                r.raise_for_status()
                return safe_json(r) or {}
            except Exception:
                return {}

    async def fetch_earnings(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        sym = (symbol or "").strip()
        if not sym:
            return []
        async with self._client(client) as c:
            try:
                r = await c.get(
                    f"{self.BASE_URL}/stock/earnings",
                    params=self._auth_params(symbol=sym, limit=limit),
                )
                r.raise_for_status()
                data = safe_json(r)
                return data if isinstance(data, list) else []
            except Exception:
                return []

    async def fetch_prices_for_symbols(
        self,
        symbols: List[str],
        client: Optional[httpx.AsyncClient] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Convenience: returns {"AAPL": 123.45, "MSFT": 0.0}
        Keeps legacy behavior but uses the same concurrency + error handling style.
        """
        if not symbols:
            return {}

        clean = [(s or "").strip() for s in symbols if (s or "").strip()]
        if not clean:
            return {}

        sem = asyncio.Semaphore(max(1, int(max_concurrency or self.max_concurrency)))

        async def fetch_one(c: httpx.AsyncClient, sym: str) -> float:
            async with sem:
                r = await c.get(f"{self.BASE_URL}/quote", params=self._auth_params(symbol=sym))
                r.raise_for_status()
                data = safe_json(r) or {}
                return float(data.get("c") or 0.0)

        async with self._client(client) as c:
            results = await asyncio.gather(
                *[fetch_one(c, s) for s in clean],
                return_exceptions=True,
            )

        out: Dict[str, float] = {}
        for sym, res in zip(clean, results):
            if isinstance(res, (Exception, BaseException)):
                out[sym] = 0.0
            else:
                out[sym] = float(res)
        return out