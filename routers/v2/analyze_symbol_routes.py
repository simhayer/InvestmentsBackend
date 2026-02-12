# routes/analysis.py
"""
FastAPI routes for AI stock analysis.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

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
async def get_full_analysis(
    symbol: str,
    include_inline: bool = Query(True, description="Include inline insights"),
    force_refresh: bool = Query(False, description="Bypass cache and recompute"),
):
    """
    Get comprehensive AI analysis for a stock.
    
    Returns cached result if less than 12h old, unless force_refresh=true.
    
    Returns full report with:
    - Investment thesis summary
    - Bull/bear cases
    - Valuation, profitability, financial health assessments
    - Risks and catalysts
    - Optional inline insights for UI badges
    """
    from services.ai.analyze_symbol.analyze_symbol_service import analyze_stock
    
    try:
        result = await analyze_stock(
            symbol.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inline/{symbol}", response_model=InlineInsightsResponse)
async def get_inline_insights(
    symbol: str,
    force_refresh: bool = Query(False, description="Bypass cache"),
):
    """
    Get quick inline insights for UI placement.
    
    Cached for 6 hours. Pass force_refresh=true to recompute.
    """
    from services.ai.analyze_symbol.analyze_symbol_service import get_stock_insights
    
    try:
        result = await get_stock_insights(symbol.upper(), force_refresh=force_refresh)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{symbol}", response_model=QuickSummaryResponse)
async def get_quick_summary(symbol: str):
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
        
        return {
            "symbol": symbol.upper(),
            "summary": result.get("summary", ""),
            "verdict": result.get("verdict", "Neutral"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{symbol}")
async def get_raw_data(symbol: str) -> Dict[str, Any]:
    """
    Get aggregated raw data without AI analysis.
    
    Useful for debugging or custom analysis.
    """
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data
    
    try:
        bundle = await aggregate_stock_data(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "data": bundle.to_dict(),
            "context": bundle.to_ai_context(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))