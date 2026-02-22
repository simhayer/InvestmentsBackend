# routes/analysis.py
"""
FastAPI routes for AI stock analysis.
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

class ValuationSection(BaseModel):
    assessment: str
    reasoning: str


class MomentumSection(BaseModel):
    earningsTrend: str
    growthTrajectory: str


class FullReportResponse(BaseModel):
    symbol: str
    summary: str
    verdict: str
    confidence: str
    valuation: ValuationSection
    profitability: ValuationSection
    financialHealth: ValuationSection
    momentum: MomentumSection
    bullCase: List[str]
    bearCase: List[str]
    risks: List[str]
    catalysts: List[str]
    technicalNotes: Optional[str] = None
    peerComparison: Optional[str] = None


class InlineInsightsResponse(BaseModel):
    valuationBadge: str
    marginCallout: str
    earningsFlag: str
    healthNote: str
    momentumSignal: str
    riskFlag: Optional[str] = None


class AnalysisResponse(BaseModel):
    symbol: str
    report: FullReportResponse
    inline: Optional[InlineInsightsResponse] = None
    dataGaps: List[str]
    cached: Optional[bool] = None
    lastAnalyzedAt: Optional[str] = None


class QuickSummaryResponse(BaseModel):
    symbol: str
    summary: str
    verdict: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/full/{symbol}", response_model=AnalysisResponse)
@limiter.limit("10/minute")
async def get_full_analysis(
    request: Request,
    symbol: str,
    include_inline: bool = Query(True, description="Include inline insights"),
    force_refresh: bool = Query(False, description="Bypass cache and recompute"),
    _user=Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive AI analysis for a stock.
    
    Returns cached result if less than 12h old, unless force_refresh=true.
    """
    require_tier(_user, db, "symbol_full_analysis")

    from services.ai.analyze_symbol.analyze_symbol_service import analyze_stock
    
    try:
        result = await analyze_stock(
            symbol.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        logger.info("symbol_analysis_completed user_id=%s symbol=%s", _user.id, symbol.upper())
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("symbol_analysis_failed user_id=%s symbol=%s: %s", _user.id, symbol, e)
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/inline/{symbol}", response_model=InlineInsightsResponse)
@limiter.limit("20/minute")
async def get_inline_insights(
    request: Request,
    symbol: str,
    force_refresh: bool = Query(False, description="Bypass cache"),
    _user=Depends(get_current_db_user),
    db: Session = Depends(get_db),
):
    """
    Get quick inline insights for UI placement.
    
    Cached for 6 hours. Pass force_refresh=true to recompute.
    """
    require_tier(_user, db, "symbol_inline")

    from services.ai.analyze_symbol.analyze_symbol_service import get_stock_insights
    
    try:
        result = await get_stock_insights(symbol.upper(), force_refresh=force_refresh)
        logger.info("symbol_inline_insights_completed user_id=%s symbol=%s", _user.id, symbol.upper())
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("symbol_inline_insights_failed user_id=%s symbol=%s: %s", _user.id, symbol, e)
        raise HTTPException(status_code=500, detail="Insights failed")


@router.get("/summary/{symbol}", response_model=QuickSummaryResponse)
@limiter.limit("20/minute")
async def get_quick_summary(
    request: Request,
    symbol: str,
    _user=Depends(get_current_db_user),
):
    """
    Get just a summary paragraph and verdict.
    
    Fastest option - minimal data fetch, simple output.
    """
    from services.ai.analyze_symbol.analyze_symbol_service import AIAnalysisService
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data
    
    try:
        bundle = await aggregate_stock_data(
            symbol.upper(),
            include_news=False,
            include_peers=False,
        )
        context = bundle.to_ai_context()
        
        ai = AIAnalysisService()
        result = await ai.generate_quick_summary(context)
        logger.info("symbol_summary_completed user_id=%s symbol=%s", _user.id, symbol.upper())
        return {
            "symbol": symbol.upper(),
            "summary": result.get("summary", ""),
            "verdict": result.get("verdict", "Neutral"),
        }
    except Exception as e:
        logger.exception("symbol_summary_failed user_id=%s symbol=%s: %s", _user.id, symbol, e)
        raise HTTPException(status_code=500, detail="Summary failed")


@router.get("/data/{symbol}")
@limiter.limit("30/minute")
async def get_raw_data(request: Request, symbol: str, _user=Depends(get_current_db_user)) -> Dict[str, Any]:
    """
    Get aggregated raw data without AI analysis.
    
    Useful for debugging or custom analysis.
    """
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data
    
    try:
        bundle = await aggregate_stock_data(symbol.upper())
        logger.info("symbol_data_fetched user_id=%s symbol=%s", _user.id, symbol.upper())
        return {
            "symbol": symbol.upper(),
            "data": bundle.to_dict(),
            "context": bundle.to_ai_context(),
        }
    except Exception as e:
        logger.exception("symbol_data_fetch_failed user_id=%s symbol=%s: %s", _user.id, symbol, e)
        raise HTTPException(status_code=500, detail="Data fetch failed")