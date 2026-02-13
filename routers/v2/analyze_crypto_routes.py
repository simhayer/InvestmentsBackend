# routers/v2/analyze_crypto_routes.py
"""
FastAPI routes for AI crypto analysis.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

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
async def get_full_crypto_analysis(
    symbol: str,
    include_inline: bool = Query(True, description="Include inline insights"),
    force_refresh: bool = Query(False, description="Bypass cache and recompute"),
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
    from services.ai.analyze_crypto.analyze_crypto_service import analyze_crypto

    try:
        result = await analyze_crypto(
            symbol.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inline/{symbol}", response_model=CryptoInlineResponse)
async def get_crypto_inline_insights(
    symbol: str,
    force_refresh: bool = Query(False, description="Bypass cache"),
):
    """
    Get quick inline insights for UI placement.

    Cached for 3 hours. Pass force_refresh=true to recompute.
    """
    from services.ai.analyze_crypto.analyze_crypto_service import get_crypto_insights

    try:
        result = await get_crypto_insights(
            symbol.upper(), force_refresh=force_refresh
        )
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{symbol}")
async def get_raw_crypto_data(symbol: str) -> Dict[str, Any]:
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
        raise HTTPException(status_code=500, detail=str(e))
