# services/binance_service.py
import httpx
import os
from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from models.crypto_asset import CryptoAsset
from services.cache.crypto_catalog import CryptoCoin, crypto_catalog
from utils.common_helpers import safe_json

BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

CRYPTOCOMPARE_COINLIST_URL = "https://min-api.cryptocompare.com/data/all/coinlist"

CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

class BinanceServiceError(Exception):
    pass


async def fetch_binance_base_assets() -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(BINANCE_EXCHANGE_INFO_URL)
            r.raise_for_status()
            data = safe_json(r) or {}

        symbols = data.get("symbols", [])
        if not isinstance(symbols, list):
            return set()

        base_assets: set[str] = set()

        for s in symbols:
            if not isinstance(s, dict):
                continue

            if s.get("status") != "TRADING":
                continue

            base = s.get("baseAsset")
            quote = s.get("quoteAsset")

            if quote not in {"USDT", "USDC", "USD"}:
                continue

            if isinstance(base, str) and base:
                base_assets.add(base)

        return base_assets

    except Exception as e:
        raise BinanceServiceError(str(e))


def upsert_crypto_assets(db: Session, provider: str, symbols: set[str]) -> int:
    existing = db.execute(
        select(CryptoAsset.symbol).where(CryptoAsset.provider == provider)
    ).scalars().all()

    existing_set = set(existing)
    to_add = symbols - existing_set

    for sym in sorted(to_add):
        db.add(CryptoAsset(symbol=sym, provider=provider, is_active=True))

    db.commit()
    return len(to_add)


def load_crypto_catalog(db: Session, provider: str = "binance") -> dict:
    rows = db.execute(
        select(CryptoAsset).where(
            CryptoAsset.provider == provider,
            CryptoAsset.is_active == True,  # noqa: E712
        )
    ).scalars().all()

    coins = [CryptoCoin(symbol=r.symbol, name=r.name) for r in rows]
    coins.sort(key=lambda c: c.symbol)
    crypto_catalog.set(coins)

    return {"provider": provider, "loaded": len(coins)}

async def fetch_crypto_metadata() -> dict[str, str]:
    """Fetches a mapping of Symbol: FullName (e.g., 'BTC': 'Bitcoin')"""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # This is a large but free public endpoint
            url = "https://min-api.cryptocompare.com/data/all/coinlist"
            r = await client.get(url)
            r.raise_for_status()
            data = r.json().get("Data", {})
            return {sym: info.get("FullName", sym) for sym, info in data.items()}
    except Exception:
        return {}
    

async def fetch_cryptocompare_symbol_names() -> dict[str, str]:
    """
    Returns mapping: {"BTC": "Bitcoin", "ETH": "Ethereum", ...}
    Uses CryptoCompare CoinList. :contentReference[oaicite:2]{index=2}
    """
    params = {}
    # CryptoCompare guide uses api_key as query param. :contentReference[oaicite:3]{index=3}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(CRYPTOCOMPARE_COINLIST_URL, params=params)
        r.raise_for_status()
        data = safe_json(r) or {}

    blob = data.get("Data")
    if not isinstance(blob, dict):
        return {}

    out: dict[str, str] = {}
    for sym, meta in blob.items():
        if not isinstance(sym, str) or not isinstance(meta, dict):
            continue

        # CryptoCompare typically has CoinName / FullName
        # FullName often looks like "Ethereum (ETH)" in many wrappers/docs.
        name = meta.get("CoinName") or meta.get("FullName") or meta.get("Name")
        if isinstance(name, str) and name.strip():
            # If FullName includes "(SYM)", keep just the readable part
            cleaned = name.strip()
            # e.g. "Ethereum (ETH)" -> "Ethereum"
            if cleaned.endswith(f"({sym})"):
                cleaned = cleaned[: cleaned.rfind("(")].strip()
            out[sym.upper()] = cleaned

    return out


def backfill_crypto_asset_names(db: Session, names: dict[str, str], provider: str = "binance") -> int:
    """
    Update DB rows where name is NULL, using the names mapping.
    Returns number of rows updated (best-effort).
    """
    if not names:
        return 0

    rows = db.execute(
        select(CryptoAsset).where(
            CryptoAsset.provider == provider,
            CryptoAsset.is_active == True,  # noqa: E712
            CryptoAsset.name.is_(None),
        )
    ).scalars().all()

    updated = 0
    for row in rows:
        nm = names.get(row.symbol.upper())
        if nm:
            row.name = nm
            updated += 1

    if updated:
        db.commit()
    return updated

async def refresh_crypto_catalog(db: Session, provider: str = "binance") -> dict:
    base_assets = await fetch_binance_base_assets()
    added = upsert_crypto_assets(db=db, provider=provider, symbols=base_assets)

    # New: fetch names + backfill
    names_map = await fetch_cryptocompare_symbol_names()
    names_updated = backfill_crypto_asset_names(db, names_map, provider=provider)

    return {
        "provider": provider,
        "coins_found": len(base_assets),
        "added": added,
        "names_updated": names_updated,
    }
