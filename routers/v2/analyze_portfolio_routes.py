# routes/portfolio_analysis.py
"""
FastAPI routes for AI portfolio analysis.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

# from core.deps import get_db, get_current_db_user, get_finnhub_service

from database import get_db
from services.supabase_auth import get_current_db_user
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService

router = APIRouter()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class DiversificationSection(BaseModel):
    assessment: str
    detail: str


class PositionInsight(BaseModel):
    symbol: str
    reasoning: Optional[str] = None
    issue: Optional[str] = None


class ActionItem(BaseModel):
    action: str  # "reduce" | "add" | "hold" | "sell" | "buy"
    symbol: Optional[str] = None
    reasoning: str


class PortfolioReportResponse(BaseModel):
    summary: str
    health: str
    riskLevel: str
    diversification: DiversificationSection
    performance: DiversificationSection
    riskExposure: DiversificationSection
    topConviction: List[PositionInsight]
    concerns: List[PositionInsight]
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    risks: List[str]
    rebalancingSuggestions: List[str]
    actionItems: List[ActionItem]


class PortfolioInlineResponse(BaseModel):
    healthBadge: str
    performanceNote: str
    riskFlag: Optional[str] = None
    topPerformer: str
    actionNeeded: Optional[str] = None


class PortfolioSummaryResponse(BaseModel):
    totalValue: float
    totalPL: float
    totalPLPct: float
    dayPL: float
    positionCount: int
    currency: str


class BenchmarkMetrics(BaseModel):
    symbol: str
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None


class SymbolRiskMetrics(BaseModel):
    volatility_annualized: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    beta: Optional[float] = None
    trading_days: Optional[int] = None


class PortfolioRiskMetrics(BaseModel):
    portfolio_beta: Optional[float] = None
    portfolio_volatility_weighted: Optional[float] = None
    hhi_concentration: Optional[float] = None
    avg_correlation_top_holdings: Optional[float] = None
    symbols_analyzed: Optional[int] = None
    per_symbol: Optional[Dict[str, SymbolRiskMetrics]] = None
    benchmark: Optional[BenchmarkMetrics] = None


class FullPortfolioAnalysisResponse(BaseModel):
    report: PortfolioReportResponse
    inline: Optional[PortfolioInlineResponse] = None
    portfolioSummary: PortfolioSummaryResponse
    riskMetrics: Optional[PortfolioRiskMetrics] = None
    dataGaps: List[str]
    cached: Optional[bool] = None
    lastAnalyzedAt: Optional[str] = None


class QuickSummaryResponse(BaseModel):
    summary: str
    health: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/full", response_model=FullPortfolioAnalysisResponse)
async def get_full_portfolio_analysis(
    currency: str = Query("USD", description="Currency for values"),
    include_inline: bool = Query(True, description="Include inline insights"),
    force_refresh: bool = Query(False, description="Bypass cache and recompute"),
    db: Session = Depends(get_db),
    user = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """
    Get comprehensive AI analysis of user's portfolio.
    
    Returns cached result if less than 24h old, unless force_refresh=true.
    
    Returns:
    - Full report with health assessment, SWOT analysis, recommendations
    - Position-level insights (top conviction, concerns)
    - Rebalancing suggestions
    - Optional inline insights for dashboard
    """
    from services.ai.portfolio.analyze_portfolio_service import analyze_portfolio
    
    try:
        result = await analyze_portfolio(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inline", response_model=PortfolioInlineResponse)
async def get_portfolio_inline_insights(
    currency: str = Query("USD"),
    db: Session = Depends(get_db),
    user = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """
    Get quick inline insights for portfolio dashboard.
    
    Faster and cheaper than full analysis.
    Returns short, punchy insights for badges/cards.
    """
    from services.ai.portfolio.analyze_portfolio_service import get_portfolio_insights
    
    try:
        result = await get_portfolio_insights(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=QuickSummaryResponse)
async def get_portfolio_summary(
    currency: str = Query("USD"),
    db: Session = Depends(get_db),
    user = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """
    Get just a summary paragraph and health assessment.
    
    Fastest option - minimal data, simple output.
    """
    from services.ai.portfolio.analyze_portfolio_aggregator import aggregate_portfolio_data
    from services.ai.portfolio.analyze_portfolio_service import AIPortfolioAnalysisService
    
    try:
        bundle = await aggregate_portfolio_data(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
            include_fundamentals=False,
            include_news=False,
        )
        context = bundle.to_ai_context()
        
        ai = AIPortfolioAnalysisService()
        result = await ai.generate_quick_summary(context)
        
        return {
            "summary": result.get("summary", ""),
            "health": result.get("health", "Good"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_portfolio_raw_data(
    currency: str = Query("USD"),
    db: Session = Depends(get_db),
    user = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
) -> Dict[str, Any]:
    """
    Get aggregated portfolio data without AI analysis.
    
    Useful for debugging or custom analysis.
    """
    from services.ai.portfolio.analyze_portfolio_aggregator import aggregate_portfolio_data
    
    try:
        bundle = await aggregate_portfolio_data(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
        )
        return {
            "data": bundle.to_dict(),
            "context": bundle.to_ai_context(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))