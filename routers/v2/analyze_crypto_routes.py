# routers/v2/analyze_crypto_routes.py
"""
FastAPI routes for AI crypto analysis.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from database import get_db
from sqlalchemy.orm import Session
from services.supabase_auth import get_current_db_user
from middleware.rate_limit import limiter
from services.tier import require_tier

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class AssessmentSection(BaseModel):
    assessment: str
    reasoning: str


class PriceActionSection(BaseModel):
    trend: str
    reasoning: str


class CryptoReportResponse(BaseModel):
    symbol: str
    summary: str
    verdict: str
    confidence: str
    marketPosition: AssessmentSection
    riskProfile: AssessmentSection
    priceAction: PriceActionSection
    bullCase: List[str]
    bearCase: List[str]
    risks: List[str]
    catalysts: List[str]
    technicalNotes: Optional[str] = None


class CryptoInlineResponse(BaseModel):
    marketCapBadge: str
    volatilityCallout: str
    trendSignal: str
    riskFlag: Optional[str] = None
    momentumNote: str


class CryptoMarketData(BaseModel):
    currentPrice: Optional[float] = None
    dayChangePct: Optional[float] = None
    marketCap: Optional[float] = None
    volume24h: Optional[float] = None
    high52w: Optional[float] = None
    low52w: Optional[float] = None


class CryptoAnalysisResponse(BaseModel):
    symbol: str
    report: CryptoReportResponse
    inline: Optional[CryptoInlineResponse] = None
    riskMetrics: Optional[Dict[str, Any]] = None
    marketData: Optional[CryptoMarketData] = None
    dataGaps: List[str]
    cached: Optional[bool] = None
    lastAnalyzedAt: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/full/{symbol}", response_model=CryptoAnalysisResponse)
@limiter.limit("10/minute")
async def get_full_crypto_analysis(
    request: Request,
    symbol: str,
    include_inline: bool = Query(True, description="Include inline insights"),
    force_refresh: bool = Query(False, description="Bypass cache and recompute"),
    _user=Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive AI analysis for a crypto asset.

    Returns cached result if less than 6h old, unless force_refresh=true.

    Returns:
    - Investment thesis summary with verdict
    - Market position, risk profile, and price action assessments
    - Bull/bear cases, risks, and catalysts
    - Quantitative risk metrics (volatility, Sharpe, max drawdown, beta vs SPY)
    - Optional inline insights for UI badges
    """
    require_tier(_user, db, "crypto_full_analysis")

    from services.ai.analyze_crypto.analyze_crypto_service import analyze_crypto

    try:
        result = await analyze_crypto(
            symbol.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Crypto analysis failed for {symbol}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/inline/{symbol}", response_model=CryptoInlineResponse)
@limiter.limit("20/minute")
async def get_crypto_inline_insights(
    request: Request,
    symbol: str,
    force_refresh: bool = Query(False, description="Bypass cache"),
    _user=Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """
    Get quick inline insights for UI placement.

    Cached for 3 hours. Pass force_refresh=true to recompute.
    """
    require_tier(_user, db, "crypto_inline")

    from services.ai.analyze_crypto.analyze_crypto_service import get_crypto_insights

    try:
        result = await get_crypto_insights(
            symbol.upper(), force_refresh=force_refresh
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Crypto inline insights failed for {symbol}")
        raise HTTPException(status_code=500, detail="Insights failed")


@router.get("/data/{symbol}")
@limiter.limit("30/minute")
async def get_raw_crypto_data(request: Request, symbol: str, _user=Depends(get_current_db_user)) -> Dict[str, Any]:
    """
    Get aggregated raw crypto data without AI analysis.

    Useful for debugging or custom analysis.
    """
    from services.ai.analyze_crypto.analyze_crypto_aggregator import aggregate_crypto_data

    try:
        bundle = await aggregate_crypto_data(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "data": bundle.to_dict(),
            "context": bundle.to_ai_context(),
        }
    except Exception as e:
        logger.exception(f"Crypto data fetch failed for {symbol}")
        raise HTTPException(status_code=500, detail="Data fetch failed")
