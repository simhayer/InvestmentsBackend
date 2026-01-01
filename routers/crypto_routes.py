# routers/crypto_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from services.binance_service import (
    refresh_crypto_catalog,
    load_crypto_catalog,
)

router = APIRouter()

@router.post("/refresh-catalog")
async def refresh_crypto_catalog_endpoint(
    db: Session = Depends(get_db),
):
    """
    Refresh crypto coin catalog from Binance.
    Safe to call manually or from a cron job.
    """

    try:
        # 1) Refresh DB from Binance
        result = await refresh_crypto_catalog(db, provider="binance")

        # 2) Reload in-memory cache FROM DB
        cache = load_crypto_catalog(db, provider="binance")

        return {
            "status": "ok",
            **result,   # provider, coins_found, added
            "cache": cache,  # provider, loaded
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh crypto catalog: {str(e)}",
        )
