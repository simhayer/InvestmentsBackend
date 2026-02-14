"""Expanded chat tool registry.

Wraps the existing ``FinnhubToolRegistry`` (Tier 1 — real-time data) and adds
higher-level analysis tools (Tier 2 — cached analysis, Tier 3 — comparison).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

from services.ai.chat.finnhub_tools import (
    FinnhubToolRegistry,
    ToolResult,
)
from services.finnhub.finnhub_service import FinnhubService

logger = logging.getLogger(__name__)


# ── Arg models for new tools ───────────────────────────────────────────

class SymbolAnalysisArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return (v or "").strip().upper()


class PortfolioAnalysisArgs(BaseModel):
    currency: str = Field(default="USD", max_length=8)

    @field_validator("currency")
    @classmethod
    def _normalize(cls, v: str) -> str:
        out = (v or "USD").strip().upper()
        return out if out in {"USD", "CAD"} else "USD"


class RiskMetricsArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return (v or "").strip().upper()


class CompareSymbolsArgs(BaseModel):
    symbols: List[str] = Field(min_length=2, max_length=5)

    @field_validator("symbols")
    @classmethod
    def _normalize(cls, v: List[str]) -> List[str]:
        return [(s or "").strip().upper() for s in v if (s or "").strip()]


# ── ChatToolRegistry ───────────────────────────────────────────────────

# Tools recognized by the new registry (superset of Finnhub tools).
ALL_TOOL_NAMES = {
    # Tier 1 — real-time data (delegated to FinnhubToolRegistry)
    "get_quote",
    "get_company_profile",
    "get_basic_financials",
    "get_peers",
    "get_portfolio_overview",
    "get_portfolio_position",
    # Tier 2 — analysis (cached, added here)
    "get_symbol_analysis",
    "get_portfolio_analysis",
    "get_risk_metrics",
    "get_portfolio_risk",
    # Tier 3 — comparison
    "compare_symbols",
}


class ChatToolRegistry:
    """Unified tool executor for the chat agent.

    * Delegates Tier-1 calls to the existing ``FinnhubToolRegistry``.
    * Adds Tier-2 analysis / risk wrappers.
    * Adds Tier-3 comparison tool.
    """

    def __init__(
        self,
        *,
        finnhub: FinnhubService,
        db: Session,
        user_id: str,
    ):
        self._finnhub = finnhub
        self._db = db
        self._user_id = user_id
        self._base = FinnhubToolRegistry(
            service=finnhub,
            db=db,
            user_id=user_id,
        )

    # ── dispatch ────────────────────────────────────────────────────

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        try:
            # Tier-2 / Tier-3 tools handled directly
            if tool_name == "get_symbol_analysis":
                parsed = SymbolAnalysisArgs(**(arguments or {}))
                return await self._get_symbol_analysis(parsed)
            if tool_name == "get_portfolio_analysis":
                parsed = PortfolioAnalysisArgs(**(arguments or {}))
                return await self._get_portfolio_analysis(parsed)
            if tool_name == "get_risk_metrics":
                parsed = RiskMetricsArgs(**(arguments or {}))
                return await self._get_risk_metrics(parsed)
            if tool_name == "get_portfolio_risk":
                return await self._get_portfolio_risk()
            if tool_name == "compare_symbols":
                parsed = CompareSymbolsArgs(**(arguments or {}))
                return await self._compare_symbols(parsed)

            # Tier-1: delegate to FinnhubToolRegistry
            return await self._base.execute(tool_name, arguments)

        except ValidationError as exc:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=f"Invalid tool arguments: {exc.errors()}",
                data_gaps=["Invalid tool arguments"],
            )
        except Exception as exc:
            logger.exception("ChatToolRegistry.execute error tool=%s", tool_name)
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=f"Tool execution failed: {exc}",
                data_gaps=["Tool execution failed"],
            )

    # ── Tier-2: analysis tools ──────────────────────────────────────

    async def _get_symbol_analysis(self, args: SymbolAnalysisArgs) -> ToolResult:
        """Full AI analysis for a stock symbol (cached 12h)."""
        from services.ai.analyze_symbol.analyze_symbol_service import analyze_stock

        result = await asyncio.wait_for(
            analyze_stock(args.symbol, include_inline=True),
            timeout=30.0,
        )
        if not result or not result.get("report"):
            return ToolResult(
                ok=False,
                tool_name="get_symbol_analysis",
                data={"symbol": args.symbol},
                data_gaps=["Analysis unavailable"],
                error="Symbol analysis unavailable",
            )

        report = result["report"]
        # Return a compact summary suitable for chat context
        data = {
            "symbol": args.symbol,
            "verdict": report.get("verdict"),
            "confidence": report.get("confidence"),
            "summary": report.get("summary"),
            "valuation": report.get("valuation"),
            "profitability": report.get("profitability"),
            "financial_health": report.get("financialHealth"),
            "momentum": report.get("momentum"),
            "bull_case": report.get("bullCase"),
            "bear_case": report.get("bearCase"),
            "risks": report.get("risks"),
            "catalysts": report.get("catalysts"),
            "cached": result.get("cached", False),
        }
        return ToolResult(ok=True, tool_name="get_symbol_analysis", data=data)

    async def _get_portfolio_analysis(self, args: PortfolioAnalysisArgs) -> ToolResult:
        """Full AI portfolio analysis (cached 24h in DB)."""
        from services.ai.portfolio.analyze_portfolio_service import analyze_portfolio

        result = await asyncio.wait_for(
            analyze_portfolio(
                self._user_id,
                self._db,
                self._finnhub,
                currency=args.currency,
                include_inline=True,
            ),
            timeout=45.0,
        )
        if not result or not result.get("report"):
            return ToolResult(
                ok=False,
                tool_name="get_portfolio_analysis",
                data={},
                data_gaps=["Portfolio analysis unavailable"],
                error="Portfolio analysis unavailable",
            )

        report = result["report"]
        data = {
            "health": report.get("health"),
            "risk_level": report.get("risk_level"),
            "summary": report.get("summary"),
            "diversification": report.get("diversification"),
            "performance": report.get("performance"),
            "strengths": report.get("strengths"),
            "weaknesses": report.get("weaknesses"),
            "opportunities": report.get("opportunities"),
            "risks": report.get("risks"),
            "action_items": report.get("action_items"),
            "concerns": report.get("concerns"),
            "cached": result.get("cached", False),
        }
        return ToolResult(ok=True, tool_name="get_portfolio_analysis", data=data)

    async def _get_risk_metrics(self, args: RiskMetricsArgs) -> ToolResult:
        """Quantitative risk metrics for a single symbol."""
        from services.ai.risk_metrics import fetch_symbol_risk_metrics

        metrics = await asyncio.wait_for(
            fetch_symbol_risk_metrics(args.symbol),
            timeout=15.0,
        )
        if not metrics:
            return ToolResult(
                ok=False,
                tool_name="get_risk_metrics",
                data={"symbol": args.symbol},
                data_gaps=["Risk metrics unavailable"],
                error="Risk metrics unavailable",
            )

        data = {"symbol": args.symbol, "metrics": metrics}
        return ToolResult(ok=True, tool_name="get_risk_metrics", data=data)

    async def _get_portfolio_risk(self) -> ToolResult:
        """Portfolio-level risk metrics (weighted beta, HHI, correlation)."""
        from services.holding_service import get_holdings_with_live_prices
        from services.ai.risk_metrics import fetch_portfolio_risk_metrics

        payload = await get_holdings_with_live_prices(
            user_id=self._user_id,
            db=self._db,
            finnhub=self._finnhub,
            currency="USD",
            top_only=False,
            top_n=15,
            include_weights=True,
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            return ToolResult(
                ok=False,
                tool_name="get_portfolio_risk",
                data={},
                data_gaps=["No holdings found"],
                error="No holdings to compute risk metrics for",
            )

        symbols = []
        weights: Dict[str, float] = {}
        for h in items:
            sym = getattr(h, "symbol", "")
            w = float(getattr(h, "weight", 0) or 0)
            if sym and w > 0:
                symbols.append(sym.upper())
                weights[sym.upper()] = w

        metrics = await asyncio.wait_for(
            fetch_portfolio_risk_metrics(symbols[:10], weights),
            timeout=20.0,
        )
        if not metrics:
            return ToolResult(
                ok=False,
                tool_name="get_portfolio_risk",
                data={},
                data_gaps=["Portfolio risk metrics unavailable"],
                error="Portfolio risk metrics unavailable",
            )

        return ToolResult(
            ok=True,
            tool_name="get_portfolio_risk",
            data={"holdings_count": len(symbols), "metrics": metrics},
        )

    # ── Tier-3: comparison ──────────────────────────────────────────

    async def _compare_symbols(self, args: CompareSymbolsArgs) -> ToolResult:
        """Fetch quotes and basic financials for multiple symbols in parallel."""
        import httpx

        async def _fetch_one(sym: str) -> Dict[str, Any]:
            price_data = await self._finnhub.get_price(sym, typ="stock")
            async with httpx.AsyncClient(timeout=5.0) as client:
                metrics_raw = await self._finnhub.fetch_basic_financials(sym, client=client)
            metric_obj = metrics_raw.get("metric", {}) if isinstance(metrics_raw, dict) else {}
            return {
                "symbol": sym,
                "currentPrice": price_data.get("currentPrice"),
                "previousClose": price_data.get("previousClose"),
                "currency": price_data.get("currency", "USD"),
                "peTTM": metric_obj.get("peTTM"),
                "pbAnnual": metric_obj.get("pbAnnual"),
                "roeTTM": metric_obj.get("roeTTM"),
                "netMargin": metric_obj.get("netMargin"),
                "52WeekHigh": metric_obj.get("52WeekHigh"),
                "52WeekLow": metric_obj.get("52WeekLow"),
            }

        results = await asyncio.gather(
            *[_fetch_one(s) for s in args.symbols],
            return_exceptions=True,
        )
        comparison = []
        errors = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(str(r))
            else:
                comparison.append(r)

        return ToolResult(
            ok=True,
            tool_name="compare_symbols",
            data={"comparison": comparison},
            data_gaps=errors if errors else [],
        )
