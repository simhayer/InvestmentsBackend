# finnhub_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from services.finnhub.finnhub_service import FinnhubService, FinnhubServiceError, format_finnhub_symbol
from cache.crypto_catalog import crypto_catalog
from services.cache.cache_backend import cache_get, cache_set, cache_get_many, cache_set_many
from services.cache.cache_utils import cacheable, should_cache_any_json
from utils.common_helpers import canonical_key
router = APIRouter()

TTL_FINNHUB_QUOTE_SEC = 60
TTL_FINNHUB_SEARCH_SEC = 15 * 60
TTL_FINNHUB_PROFILE_SEC = 24 * 60 * 60

def _ck(kind: str, *parts: str) -> str:
    """Safe cache key builder."""
    clean = [p.strip() for p in parts if p and p.strip()]
    return f"finnhub:{kind}:" + ":".join(clean)

def get_finnhub_service() -> FinnhubService:
    return FinnhubService()

# ---------- Schemas ----------
class PriceRequest(BaseModel):
    symbols: List[str]
    types: List[str]
    currency: str = "USD"

# -------------------------
# Routes
# -------------------------

@router.get("/price")
async def get_price(
    symbol: str,
    type: str = "stock",
    svc: FinnhubService = Depends(get_finnhub_service),
):
    sym = (symbol or "").strip()
    typ = (type or "stock").strip()

    if not sym:
        raise HTTPException(status_code=400, detail="Missing symbol")

    # cache by the formatted symbol (important for crypto like BINANCE:BTCUSDT)
    formatted = format_finnhub_symbol(sym, typ)
    cache_key = _ck("quote", formatted)

    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = await svc.get_price(symbol=sym, typ=typ)
        cache_set(cache_key, payload, ttl_seconds=TTL_FINNHUB_QUOTE_SEC)
        return payload
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching price: {e}")


@router.post("/prices")
async def get_prices(
    request: PriceRequest,
    svc: FinnhubService = Depends(get_finnhub_service),
):
    if len(request.symbols) != len(request.types):
        raise HTTPException(status_code=400, detail="Symbols and types list length mismatch")

    pairs: List[Tuple[str, str]] = []
    for s, t in zip(request.symbols, request.types):
        s2 = (s or "").strip()
        t2 = (t or "stock").strip()
        if s2:
            pairs.append((s2, t2))

    if not pairs:
        return {}

    # Cache per formatted symbol
    formatted_list = [format_finnhub_symbol(sym, typ) for sym, typ in pairs]
    keys = [_ck("quote", fs) for fs in formatted_list]

    cached_map = cache_get_many(keys)
    out: Dict[str, Any] = {}
    misses: List[Tuple[str, str, str, str]] = []  # (orig_sym, typ, formatted, cache_key)

    for (sym, typ), fs, k in zip(pairs, formatted_list, keys):
        hit = cached_map.get(k)
        if hit is not None:
            out_key = canonical_key(sym, typ)
            out[out_key] = hit
        else:
            misses.append((sym, typ, fs, k))

    if misses:
        try:
            # Only fetch misses
            miss_pairs = [(sym, typ) for sym, typ, _, _ in misses]
            fresh = await svc.get_prices(pairs=miss_pairs, currency=request.currency)

            # Store each miss response in cache
            write_back: Dict[str, Any] = {}

            # svc.get_prices returns keys as canonical_key(sym, typ)
            for (sym, typ, fs, ck) in misses:
                canon = f"{sym}:{typ}"
                payload = fresh.get(canon)
                out[canon] = payload

                write_back[ck] = payload

            cache_set_many(write_back, ttl_seconds=TTL_FINNHUB_QUOTE_SEC)

        except FinnhubServiceError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching prices: {e}")

    return out


def _search_key(query: str) -> str:
    q = (query or "").strip().lower()
    return _ck("search", q)

@cacheable(ttl=TTL_FINNHUB_SEARCH_SEC, key_fn=lambda query, svc: _search_key(query), should_cache=should_cache_any_json)
async def _search_symbols_cached(query: str, svc: FinnhubService) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    limit = 5
    merged: List[Dict[str, Any]] = []

    # 1) Crypto first
    crypto_hits = crypto_catalog.search(q, limit=limit)
    for c in crypto_hits:
        merged.append({
            "symbol": c.symbol,
            "description": f"{c.name} ({c.symbol}) • Crypto" if c.name else f"{c.symbol} • Crypto",
            "asset_type": "crypto",
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
                if len(merged) >= limit * 3:
                    break
    except Exception:
        pass

    # 3) De-dupe by SYMBOL
    seen = set()
    final: List[Dict[str, Any]] = []
    for item in merged:
        sym = item.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        final.append(item)
        if len(final) >= limit:
            break

    return final


@router.get("/search")
async def search_symbols(
    query: str,
    svc: FinnhubService = Depends(get_finnhub_service),
):
    return await _search_symbols_cached(query, svc)

@router.get("/quote")
async def fetch_quote(
    symbol: str,
    svc: FinnhubService = Depends(get_finnhub_service),
):
    sym = (symbol or "").strip()
    if not sym:
        raise HTTPException(status_code=400, detail="Missing symbol")

    cache_key = _ck("rawquote", sym.upper())
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = await svc.fetch_quote(sym)
        cache_set(cache_key, payload, ttl_seconds=TTL_FINNHUB_QUOTE_SEC)
        return payload
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profile")
async def fetch_profile(
    symbol: str,
    svc: FinnhubService = Depends(get_finnhub_service),
):
    sym = (symbol or "").strip()
    if not sym:
        raise HTTPException(status_code=400, detail="Missing symbol")

    cache_key = _ck("profile", sym.upper())
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = await svc.fetch_profile(sym)
        cache_set(cache_key, payload, ttl_seconds=TTL_FINNHUB_PROFILE_SEC)
        return payload
    except FinnhubServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
