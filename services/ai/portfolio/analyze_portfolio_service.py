# ai_portfolio_analysis.py
"""
AI-powered portfolio analysis service.
Generates comprehensive portfolio reviews, risk assessments, and recommendations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

from services.ai.llm_service import get_llm_service


class PortfolioHealth(str, Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    NEEDS_ATTENTION = "Needs Attention"


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


@dataclass
class PortfolioAnalysisReport:
    """Structured portfolio analysis report."""

    # Overview
    summary: str
    health: PortfolioHealth
    risk_level: RiskLevel

    # Key metrics assessment
    diversification: Dict[str, str]  # assessment, detail
    performance: Dict[str, str]      # assessment, detail
    risk_exposure: Dict[str, str]    # assessment, detail

    # Position insights
    top_conviction: List[Dict[str, str]]   # symbol, reasoning
    concerns: List[Dict[str, str]]          # symbol, issue

    # Actionable insights
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    risks: List[str]

    # Recommendations
    rebalancing_suggestions: List[str]
    action_items: List[Dict[str, Any]]  # structured: {action, symbol, reasoning}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "health": self.health.value,
            "riskLevel": self.risk_level.value,
            "diversification": self.diversification,
            "performance": self.performance,
            "riskExposure": self.risk_exposure,
            "topConviction": self.top_conviction,
            "concerns": self.concerns,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "opportunities": self.opportunities,
            "risks": self.risks,
            "rebalancingSuggestions": self.rebalancing_suggestions,
            "actionItems": self.action_items,
        }


@dataclass
class PortfolioInlineInsights:
    """Quick inline insights for dashboard display."""
    health_badge: str            # "Well Diversified" / "Concentrated"
    performance_note: str        # "+12.5% overall, beating S&P"
    risk_flag: Optional[str]     # "High tech exposure" (only if notable)
    top_performer: str           # "NVDA +145%"
    action_needed: Optional[str] # "Consider rebalancing" (only if needed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthBadge": self.health_badge,
            "performanceNote": self.performance_note,
            "riskFlag": self.risk_flag,
            "topPerformer": self.top_performer,
            "actionNeeded": self.action_needed,
        }


# ============================================================================
# PROMPTS
# ============================================================================

PORTFOLIO_SYSTEM_PROMPT = """You are a senior portfolio analyst at a wealth-management firm.
Your role is to provide data-driven, quantitative portfolio analysis personalized to the investor.

Guidelines:
- ALWAYS cite specific numbers: beta values, Sharpe ratios, volatility percentages, HHI scores, correlations, P/L figures, and portfolio weights
- When benchmark data (S&P 500) is provided, explicitly compare portfolio metrics against it (e.g. "Portfolio Sharpe of 0.85 vs S&P 500 Sharpe of 1.12")
- Reference concentration risk using HHI scores and top-3 weight percentages
- Use risk-adjusted return metrics (Sharpe, Sortino) to evaluate performance — not just raw returns
- Be specific about sector/industry exposure using the industry data provided
- Give actionable, position-level recommendations with target weights when suggesting rebalancing
- Consider correlation structure — flag when holdings are highly correlated (avg correlation > 0.7)

Personalization (when Investor Profile is provided):
- Calibrate risk assessments to the investor's stated risk tolerance — a "High" risk portfolio may be appropriate for an aggressive investor but alarming for a conservative one
- Match recommendation complexity to experience level: plain language for beginners, technical detail for advanced investors
- Align suggestions with the investor's primary goal (growth vs income vs preservation)
- Consider time horizon when evaluating drawdowns and volatility — a long-term investor can tolerate more short-term volatility
- Respect stated asset preferences — don't recommend crypto to someone who excluded it, or vice versa
- If the portfolio contradicts the investor's profile (e.g. conservative investor with speculative positions), flag this mismatch explicitly

You always respond with valid JSON matching the requested schema."""


PORTFOLIO_FULL_REPORT_PROMPT = """Analyze the following portfolio and provide a comprehensive, data-driven review.

{context}

Respond with a JSON object matching this exact schema:
{{
    "summary": "2-3 sentence assessment. MUST cite key numbers: portfolio beta, Sharpe ratio, total return vs benchmark if available.",
    "health": "Excellent" | "Good" | "Fair" | "Needs Attention",
    "riskLevel": "Low" | "Moderate" | "High" | "Very High",

    "diversification": {{
        "assessment": "Well Diversified" | "Adequate" | "Concentrated" | "Highly Concentrated",
        "detail": "Cite HHI score, top-3 weight %, sector breakdown, and avg correlation between holdings."
    }},

    "performance": {{
        "assessment": "Strong" | "Moderate" | "Weak" | "Mixed",
        "detail": "Cite total P/L, total return %, and compare to S&P 500 benchmark if data is available. Reference Sharpe/Sortino ratios."
    }},

    "riskExposure": {{
        "assessment": "Conservative" | "Balanced" | "Aggressive" | "Speculative",
        "detail": "Cite portfolio beta, weighted volatility, max drawdown, and any correlation concerns."
    }},

    "topConviction": [
        {{"symbol": "AAPL", "reasoning": "Reference P/L, Sharpe ratio, beta, or fundamentals that make this a strong hold"}}
    ],

    "concerns": [
        {{"symbol": "XYZ", "issue": "Reference specific metrics: loss amount, high beta, poor Sharpe, overweight %, etc."}}
    ],

    "strengths": ["Cite specific data points in each strength"],
    "weaknesses": ["Cite specific data points in each weakness"],
    "opportunities": ["Specific opportunity with rationale"],
    "risks": ["Specific risk with quantification when possible"],

    "rebalancingSuggestions": [
        "Specific rebalancing recommendation referencing current vs target weights"
    ],

    "actionItems": [
        {{
            "action": "reduce" | "add" | "hold" | "sell" | "buy",
            "symbol": "NVDA",
            "reasoning": "Specific rationale citing data (e.g. 'Beta of 1.8 at 25% weight contributes disproportionate portfolio risk')"
        }}
    ]
}}

Critical instructions:
- EVERY observation must reference actual numbers from the data — never make vague claims
- topConviction: 2-3 positions with the best risk-adjusted profile (cite Sharpe, P/L, beta)
- concerns: only positions with measurable issues (cite losses, high beta, low Sharpe, overweight)
- actionItems: specific position-level recommendations with action type and symbol
- When benchmark data is available, the summary and performance MUST compare against it
- If HHI > 2500 or top-3 weight > 60%, flag concentration risk explicitly
- If avg correlation > 0.7, flag diversification concern
- If portfolio is small (<5 positions), note limited diversification
- If an Investor Profile is provided, personalize the tone and recommendations:
  * For beginners: use plain language, explain financial terms briefly
  * For advanced investors: use technical language freely
  * Flag any mismatch between the portfolio and the investor's stated risk level or goals
  * Align rebalancing suggestions with the investor's time horizon and primary goal"""


PORTFOLIO_INLINE_PROMPT = """Based on the following portfolio data, generate brief quantitative insights for dashboard display.
Each insight should cite specific numbers — never be vague.

{context}

Respond with a JSON object:
{{
    "healthBadge": "Short status with a number, e.g., 'HHI 1200 — Well Diversified' or 'Beta 1.4 — Aggressive'",
    "performanceNote": "Return with benchmark comparison if available, e.g., '+18.5% vs SPY +22%' or 'Sharpe 1.2, beating benchmark'",
    "riskFlag": "Cite a specific risk metric if concerning (e.g., 'Top 3 = 72% of portfolio, HHI 3100'), or null if balanced",
    "topPerformer": "Best position with data, e.g., 'NVDA +145%, Sharpe 2.1'",
    "actionNeeded": "Specific action with reason (e.g., 'NVDA at 25% — trim to 15% to reduce beta'), or null"
}}

Be specific with numbers. Use null for riskFlag and actionNeeded if nothing notable.
If an Investor Profile is provided, calibrate the actionNeeded urgency to their risk tolerance and goals."""


PORTFOLIO_QUICK_SUMMARY_PROMPT = """Provide a one-paragraph portfolio summary.
Be direct about portfolio health and any immediate concerns or opportunities.

{context}

Respond with JSON:
{{
    "summary": "3-4 sentence summary covering health, performance, and key observations",
    "health": "Excellent" | "Good" | "Fair" | "Needs Attention"
}}"""


# ============================================================================
# SERVICE
# ============================================================================

class AIPortfolioAnalysisService:
    """Generates AI-powered portfolio analysis."""

    def __init__(
        self,
    ):
        self.llm = get_llm_service()

    async def _call_llm(self, system: str, user: str) -> str:
        """
        Call LLM provider.
        Now uses shared LLM service (provider/model selected via env in llm_service).
        """
        data = await self.llm.generate_json(system=system, user=user)

        # llm_service returns dict, but this file expects a JSON string at this layer.
        # Return a stable JSON string so downstream parsing behaves the same.
        return json.dumps(data)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return json.loads(text)

    async def generate_full_report(
        self,
        context: str,
    ) -> PortfolioAnalysisReport:
        """Generate comprehensive portfolio analysis."""
        prompt = PORTFOLIO_FULL_REPORT_PROMPT.format(context=context)
        response = await self._call_llm(PORTFOLIO_SYSTEM_PROMPT, prompt)
        data = self._parse_json(response)

        return PortfolioAnalysisReport(
            summary=data.get("summary", ""),
            health=PortfolioHealth(data.get("health", "Good")),
            risk_level=RiskLevel(data.get("riskLevel", "Moderate")),
            diversification=data.get("diversification", {}),
            performance=data.get("performance", {}),
            risk_exposure=data.get("riskExposure", {}),
            top_conviction=data.get("topConviction", []),
            concerns=data.get("concerns", []),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            opportunities=data.get("opportunities", []),
            risks=data.get("risks", []),
            rebalancing_suggestions=data.get("rebalancingSuggestions", []),
            action_items=data.get("actionItems", []),
        )

    async def generate_inline_insights(
        self,
        context: str,
    ) -> PortfolioInlineInsights:
        """Generate quick inline insights for dashboard."""
        prompt = PORTFOLIO_INLINE_PROMPT.format(context=context)
        response = await self._call_llm(PORTFOLIO_SYSTEM_PROMPT, prompt)
        data = self._parse_json(response)

        return PortfolioInlineInsights(
            health_badge=data.get("healthBadge", ""),
            performance_note=data.get("performanceNote", ""),
            risk_flag=data.get("riskFlag"),
            top_performer=data.get("topPerformer", ""),
            action_needed=data.get("actionNeeded"),
        )

    async def generate_quick_summary(
        self,
        context: str,
    ) -> Dict[str, str]:
        """Generate just a summary and health assessment."""
        prompt = PORTFOLIO_QUICK_SUMMARY_PROMPT.format(context=context)
        response = await self._call_llm(PORTFOLIO_SYSTEM_PROMPT, prompt)
        return self._parse_json(response)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def analyze_portfolio(
    user_id: str,
    db,
    finnhub,
    *,
    currency: str = "USD",
    include_inline: bool = True,
) -> Dict[str, Any]:
    """
    Full portfolio analysis pipeline.

    Usage:
        result = await analyze_portfolio(user_id, db, finnhub)
        print(result["report"]["summary"])
    """
    from services.ai.portfolio.analyze_portfolio_aggregator import aggregate_portfolio_data

    # Gather data
    bundle = await aggregate_portfolio_data(
        user_id, db, finnhub,
        currency=currency,
        include_fundamentals=True,
        include_news=True,
    )
    context = bundle.to_ai_context()

    # Initialize AI service (provider/model configured via env in llm_service)
    ai = AIPortfolioAnalysisService()

    # Generate report
    report = await ai.generate_full_report(context)

    result = {
        "report": report.to_dict(),
        "portfolioSummary": {
            "totalValue": bundle.total_value,
            "totalPL": bundle.total_pl,
            "totalPLPct": bundle.total_pl_pct,
            "dayPL": bundle.day_pl,
            "positionCount": bundle.position_count,
            "currency": bundle.currency,
        },
        "riskMetrics": bundle.risk_metrics if bundle.risk_metrics else None,
        "dataGaps": bundle.gaps,
    }

    # Generate inline insights if requested
    if include_inline:
        inline = await ai.generate_inline_insights(context)
        result["inline"] = inline.to_dict()

    return result


async def get_portfolio_insights(
    user_id: str,
    db,
    finnhub,
    currency: str = "USD",
) -> Dict[str, str]:
    """
    Quick inline insights only (faster, cheaper).
    """
    from services.ai.portfolio.analyze_portfolio_aggregator import aggregate_portfolio_data

    bundle = await aggregate_portfolio_data(
        user_id, db, finnhub,
        currency=currency,
        include_fundamentals=False,
        include_news=False,
    )
    context = bundle.to_ai_context()

    ai = AIPortfolioAnalysisService()
    inline = await ai.generate_inline_insights(context)

    return inline.to_dict()
