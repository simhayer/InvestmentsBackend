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
    action_items: List[str]

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

PORTFOLIO_SYSTEM_PROMPT = """You are a professional portfolio analyst and financial advisor. 
Your role is to provide clear, actionable portfolio analysis based on the data provided.

Guidelines:
- Be direct and specific with observations
- Support claims with data from the portfolio
- Focus on actionable insights, not generic advice
- Consider both opportunities and risks
- Be balanced - acknowledge both strengths and weaknesses
- Tailor advice to the portfolio size and composition

You always respond with valid JSON matching the requested schema."""


PORTFOLIO_FULL_REPORT_PROMPT = """Analyze the following portfolio and provide a comprehensive review.

{context}

Respond with a JSON object matching this exact schema:
{{
    "summary": "2-3 sentence portfolio assessment",
    "health": "Excellent" | "Good" | "Fair" | "Needs Attention",
    "riskLevel": "Low" | "Moderate" | "High" | "Very High",

    "diversification": {{
        "assessment": "Well Diversified" | "Adequate" | "Concentrated" | "Highly Concentrated",
        "detail": "1-2 sentences on sector/position concentration"
    }},

    "performance": {{
        "assessment": "Strong" | "Moderate" | "Weak" | "Mixed",
        "detail": "1-2 sentences on overall P/L and notable performers"
    }},

    "riskExposure": {{
        "assessment": "Conservative" | "Balanced" | "Aggressive" | "Speculative",
        "detail": "1-2 sentences on risk factors"
    }},

    "topConviction": [
        {{"symbol": "AAPL", "reasoning": "Why this is a strong position"}},
        {{"symbol": "MSFT", "reasoning": "Why this is a strong position"}}
    ],

    "concerns": [
        {{"symbol": "XYZ", "issue": "What's concerning about this position"}},
        {{"symbol": "ABC", "issue": "What's concerning"}}
    ],

    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "opportunities": ["opportunity 1", "opportunity 2"],
    "risks": ["risk 1", "risk 2", "risk 3"],

    "rebalancingSuggestions": [
        "Specific rebalancing recommendation 1",
        "Specific rebalancing recommendation 2"
    ],

    "actionItems": [
        "Prioritized action 1",
        "Prioritized action 2"
    ]
}}

Important:
- Base all observations on the actual data provided
- topConviction should highlight 2-3 strongest positions with reasoning
- concerns should only include positions with real issues (losses, overweight, etc.)
- rebalancingSuggestions should be specific and actionable
- If portfolio is small (<5 positions), note limited diversification
- Consider position sizes when assessing risk"""


PORTFOLIO_INLINE_PROMPT = """Based on the following portfolio data, generate brief insights for dashboard display.
Each insight should be concise and impactful.

{context}

Respond with a JSON object:
{{
    "healthBadge": "Short health status, e.g., 'Well Diversified' or '15 positions, balanced'",
    "performanceNote": "Brief P/L summary, e.g., '+18.5% YTD' or 'Up $12,450 overall'",
    "riskFlag": "Notable risk if any (e.g., '60% in tech'), or null if portfolio is balanced",
    "topPerformer": "Best position, e.g., 'NVDA +145%'",
    "actionNeeded": "Urgent action if needed (e.g., 'Consider taking profits on NVDA'), or null"
}}

Be specific with numbers. Use null for riskFlag and actionNeeded if nothing notable."""


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
