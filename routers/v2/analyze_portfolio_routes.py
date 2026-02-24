# routes/portfolio_analysis.py
"""
FastAPI routes for AI portfolio analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database import get_db
from services.supabase_auth import get_current_db_user
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService
from middleware.rate_limit import limiter
from services.tier import require_tier, increment_usage, get_user_plan
from models.portfolio_analysis import PortfolioAnalysis

logger = logging.getLogger(__name__)

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
    stale: Optional[bool] = None
    lastAnalyzedAt: Optional[str] = None


class QuickSummaryResponse(BaseModel):
    summary: str
    health: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/full", response_model=FullPortfolioAnalysisResponse)
@limiter.limit("5/minute")
async def get_full_portfolio_analysis(
    request: Request,
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

    # Helper: load cached row from DB
    def _get_cached():
        try:
            return db.query(PortfolioAnalysis).filter(
                PortfolioAnalysis.user_id == int(user.id)
            ).first()
        except Exception:
            logger.exception("portfolio_cache_check_failed user_id=%s", user.id)
            return None

    def _row_to_response(row, *, stale: bool = False):
        row.data["cached"] = True
        row.data["stale"] = stale
        row.data["lastAnalyzedAt"] = row.created_at.isoformat()
        return row.data

    # 1) Fresh cache — serve without consuming quota
    if not force_refresh:
        cached_row = _get_cached()
        if cached_row and cached_row.data:
            age_hours = (
                datetime.now(timezone.utc)
                - cached_row.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 3600
            if age_hours < 24:
                logger.info("portfolio_analysis_cache_hit user_id=%s age_h=%.1f", user.id, age_hours)
                return _row_to_response(cached_row)

    # 2) Stale/missing cache — check quota (without incrementing)
    try:
        require_tier(user, db, "portfolio_full_analysis", increment=False)
    except HTTPException as tier_exc:
        if tier_exc.status_code == 403:
            stale_row = _get_cached()
            if stale_row and stale_row.data:
                logger.info("portfolio_analysis_stale_fallback user_id=%s", user.id)
                return _row_to_response(stale_row, stale=True)
        raise

    # 3) Run fresh analysis, only increment quota on success
    try:
        result = await analyze_portfolio(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
            include_inline=include_inline,
            force_refresh=force_refresh,
        )
        plan = get_user_plan(user, db)
        increment_usage(user.id, "portfolio_full_analysis", plan)
        logger.info("portfolio_analysis_completed user_id=%s currency=%s", user.id, currency.upper())
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("portfolio_analysis_failed user_id=%s: %s", user.id, e)
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/inline", response_model=PortfolioInlineResponse)
@limiter.limit("10/minute")
async def get_portfolio_inline_insights(
    request: Request,
    currency: str = Query("USD"),
    force_refresh: bool = Query(False, description="Bypass inline cache"),
    db: Session = Depends(get_db),
    user = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """
    Get quick inline insights for portfolio dashboard.
    
    Cached in Redis for 3 hours per user. Pass force_refresh=true to recompute.
    Returns short, punchy insights for badges/cards.
    """
    from services.ai.portfolio.analyze_portfolio_service import get_portfolio_insights
    from services.cache.cache_backend import cache_get as _cache_get

    # Serve cached inline insights without consuming quota
    if not force_refresh:
        try:
            cache_key = f"portfolio:inline:{user.id}:{currency.upper()}"
            cached = _cache_get(cache_key)
            if cached and isinstance(cached, dict):
                logger.info("portfolio_inline_cache_hit user_id=%s", user.id)
                return cached
        except Exception:
            logger.exception("portfolio_inline_cache_check_failed user_id=%s", user.id)

    require_tier(user, db, "portfolio_inline")

    try:
        result = await get_portfolio_insights(
            str(user.id),
            db,
            finnhub,
            currency=currency.upper(),
            force_refresh=force_refresh,
        )
        logger.info("portfolio_inline_insights_completed user_id=%s", user.id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("portfolio_inline_insights_failed user_id=%s: %s", user.id, e)
        raise HTTPException(status_code=500, detail="Insights failed")


@router.get("/summary", response_model=QuickSummaryResponse)
@limiter.limit("10/minute")
async def get_portfolio_summary(
    request: Request,
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
        logger.info("portfolio_summary_completed user_id=%s", user.id)
        return {
            "summary": result.get("summary", ""),
            "health": result.get("health", "Good"),
        }
    except Exception as e:
        logger.exception("portfolio_summary_failed user_id=%s: %s", user.id, e)
        raise HTTPException(status_code=500, detail="Summary failed")


@router.get("/data")
@limiter.limit("15/minute")
async def get_portfolio_raw_data(
    request: Request,
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
        logger.info("portfolio_data_fetched user_id=%s", user.id)
        return {
            "data": bundle.to_dict(),
            "context": bundle.to_ai_context(),
        }
    except Exception as e:
        logger.exception("portfolio_data_fetch_failed user_id=%s: %s", user.id, e)
        raise HTTPException(status_code=500, detail="Data fetch failed")