# ai_stock_analysis.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from services.ai.llm_service import get_llm_service
from services.cache.cache_backend import cache_get, cache_set

logger = logging.getLogger(__name__)

# Cache TTL for symbol analysis (12 hours in seconds)
SYMBOL_ANALYSIS_TTL_SEC = 12 * 3600
SYMBOL_INLINE_TTL_SEC = 6 * 3600

def _analysis_cache_key(symbol: str) -> str:
    return f"analysis:full:{symbol.upper()}"

def _inline_cache_key(symbol: str) -> str:
    return f"analysis:inline:{symbol.upper()}"


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class Verdict(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class AnalysisReport:
    symbol: str
    summary: str
    verdict: Verdict
    confidence: Confidence
    valuation: Dict[str, str]
    profitability: Dict[str, str]
    financial_health: Dict[str, str]
    momentum: Dict[str, str]
    bull_case: List[str]
    bear_case: List[str]
    risks: List[str]
    catalysts: List[str]
    technical_notes: Optional[str] = None
    peer_comparison: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "summary": self.summary,
            "verdict": self.verdict.value,
            "confidence": self.confidence.value,
            "valuation": self.valuation,
            "profitability": self.profitability,
            "financialHealth": self.financial_health,
            "momentum": self.momentum,
            "bullCase": self.bull_case,
            "bearCase": self.bear_case,
            "risks": self.risks,
            "catalysts": self.catalysts,
            "technicalNotes": self.technical_notes,
            "peerComparison": self.peer_comparison,
        }


@dataclass
class InlineInsights:
    valuation_badge: str
    margin_callout: str
    earnings_flag: str
    health_note: str
    momentum_signal: str
    risk_flag: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valuationBadge": self.valuation_badge,
            "marginCallout": self.margin_callout,
            "earningsFlag": self.earnings_flag,
            "healthNote": self.health_note,
            "momentumSignal": self.momentum_signal,
            "riskFlag": self.risk_flag,
        }


# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are a professional equity research analyst. Your role is to provide clear,
actionable stock analysis based on fundamental data.

Guidelines:
- Be direct and opinionated - take a stance
- Support claims with specific data points
- Acknowledge data gaps honestly
- Use plain language, avoid jargon
- Focus on what matters for investment decisions
- Be concise but thorough

You always respond with valid JSON matching the requested schema.
Return ONLY JSON. No markdown. No extra text.
"""

FULL_REPORT_PROMPT = """Analyze the following stock data and provide a comprehensive investment analysis.

{context}

Respond with a JSON object matching this exact schema:
{{
    "summary": "2-3 sentence investment thesis",
    "verdict": "Bullish" | "Bearish" | "Neutral",
    "confidence": "High" | "Medium" | "Low",
    "valuation": {{
        "assessment": "Cheap" | "Fair" | "Expensive",
        "reasoning": "1-2 sentences explaining why"
    }},
    "profitability": {{
        "assessment": "Strong" | "Moderate" | "Weak",
        "reasoning": "1-2 sentences"
    }},
    "financialHealth": {{
        "assessment": "Solid" | "Adequate" | "Concerning",
        "reasoning": "1-2 sentences"
    }},
    "momentum": {{
        "earningsTrend": "Beating" | "Mixed" | "Missing",
        "growthTrajectory": "Accelerating" | "Stable" | "Decelerating"
    }},
    "bullCase": ["point 1", "point 2", "point 3"],
    "bearCase": ["point 1", "point 2", "point 3"],
    "risks": ["risk 1", "risk 2", "risk 3"],
    "catalysts": ["catalyst 1", "catalyst 2"],
    "technicalNotes": "1-2 sentences on price position vs 52-week range and moving averages, or null if no data",
    "peerComparison": "1-2 sentences comparing to the listed peers, or null if no peers listed"
}}
"""

INLINE_INSIGHTS_PROMPT = """Based on the following stock data, generate brief inline insights for UI display.
Each insight should be punchy and under 50 characters.

{context}

Respond with a JSON object:
{{
    "valuationBadge": "e.g., 'Trading at 15x vs 22x sector avg'",
    "marginCallout": "e.g., 'Operating margins up 200bps YoY'",
    "earningsFlag": "e.g., 'Beat estimates 4/4 quarters'",
    "healthNote": "e.g., 'Net cash position, no debt'",
    "momentumSignal": "e.g., 'Revenue growth accelerating'",
    "riskFlag": "Notable risk if any, or null"
}}
"""

QUICK_SUMMARY_PROMPT = """Provide a one-paragraph investment summary for this stock.
Be direct about whether this is an attractive investment and why.

{context}

Respond with JSON:
{{
    "summary": "3-4 sentence summary with clear stance",
    "verdict": "Bullish" | "Bearish" | "Neutral"
}}
"""


# ============================================================================
# SERVICE (CALLER NEVER CHOOSES PROVIDER)
# ============================================================================

class AIAnalysisService:
    """
    Public API stays stable.
    Internally selects the provider/model based on env/config only.
    """

    def __init__(self):
        self.llm = get_llm_service()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        # Keep this here to avoid changing other behavior/assumptions.
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def _safe_enum(self, enum_cls, value: str, default):
        try:
            return enum_cls(value)
        except Exception:
            return default

    async def _ask(self, prompt: str) -> Dict[str, Any]:
        # New shared LLM service handles provider selection.
        raw_obj = await self.llm.generate_json(system=SYSTEM_PROMPT, user=prompt)

        # llm_service.generate_json returns dict already, but keep compatibility:
        if isinstance(raw_obj, dict):
            return raw_obj
        return self._parse_json(str(raw_obj))

    async def generate_full_report(self, context: str, symbol: str) -> AnalysisReport:
        data = await self._ask(FULL_REPORT_PROMPT.format(context=context))

        verdict = self._safe_enum(Verdict, data.get("verdict", "Neutral"), Verdict.NEUTRAL)
        conf = self._safe_enum(Confidence, data.get("confidence", "Medium"), Confidence.MEDIUM)

        return AnalysisReport(
            symbol=symbol,
            summary=data.get("summary", ""),
            verdict=verdict,
            confidence=conf,
            valuation=data.get("valuation", {}) or {},
            profitability=data.get("profitability", {}) or {},
            financial_health=data.get("financialHealth", {}) or {},
            momentum=data.get("momentum", {}) or {},
            bull_case=data.get("bullCase", []) or [],
            bear_case=data.get("bearCase", []) or [],
            risks=data.get("risks", []) or [],
            catalysts=data.get("catalysts", []) or [],
            technical_notes=data.get("technicalNotes"),
            peer_comparison=data.get("peerComparison"),
        )

    async def generate_inline_insights(self, context: str) -> InlineInsights:
        data = await self._ask(INLINE_INSIGHTS_PROMPT.format(context=context))
        return InlineInsights(
            valuation_badge=data.get("valuationBadge", "") or "",
            margin_callout=data.get("marginCallout", "") or "",
            earnings_flag=data.get("earningsFlag", "") or "",
            health_note=data.get("healthNote", "") or "",
            momentum_signal=data.get("momentumSignal", "") or "",
            risk_flag=data.get("riskFlag"),
        )

    async def generate_quick_summary(self, context: str) -> Dict[str, str]:
        data = await self._ask(QUICK_SUMMARY_PROMPT.format(context=context))
        verdict = data.get("verdict", "Neutral") or "Neutral"
        return {"summary": data.get("summary", "") or "", "verdict": verdict}


# ============================================================================
# CONVENIENCE FUNCTIONS (caller still doesn't choose provider)
# ============================================================================

async def analyze_stock(
    symbol: str,
    *,
    include_inline: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Full stock analysis with Redis caching.
    
    Cache TTL: 12 hours. Symbol analysis is not user-specific,
    so all users share the same cached result for a given ticker.
    """
    sym = symbol.upper()
    cache_key = _analysis_cache_key(sym)

    # Check cache first
    if not force_refresh:
        cached = cache_get(cache_key)
        if cached and isinstance(cached, dict):
            logger.info(f"[Symbol Analysis] cache hit for {sym}")
            cached["cached"] = True
            return cached

    # Compute fresh analysis
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data

    bundle = await aggregate_stock_data(sym)
    context = bundle.to_ai_context()

    ai = AIAnalysisService()
    report = await ai.generate_full_report(context, sym)

    result: Dict[str, Any] = {
        "symbol": sym,
        "report": report.to_dict(),
        "dataGaps": bundle.gaps,
    }

    if include_inline:
        inline = await ai.generate_inline_insights(context)
        result["inline"] = inline.to_dict()

    result["lastAnalyzedAt"] = datetime.now(timezone.utc).isoformat()

    # Persist to Redis
    try:
        cache_set(cache_key, result, ttl_seconds=SYMBOL_ANALYSIS_TTL_SEC)
        logger.info(f"[Symbol Analysis] cached result for {sym}")
    except Exception as e:
        logger.warning(f"[Symbol Analysis] failed to cache: {e}")

    result["cached"] = False
    return result


async def get_stock_insights(
    symbol: str,
    *,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """
    Inline insights with Redis caching(6h TTL).
    """
    sym = symbol.upper()
    cache_key = _inline_cache_key(sym)

    if not force_refresh:
        cached = cache_get(cache_key)
        if cached and isinstance(cached, dict):
            logger.info(f"[Symbol Inline] cache hit for {sym}")
            return cached

    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data

    bundle = await aggregate_stock_data(sym, include_news=False)
    context = bundle.to_ai_context()

    ai = AIAnalysisService()
    inline = await ai.generate_inline_insights(context)
    result = inline.to_dict()

    try:
        cache_set(cache_key, result, ttl_seconds=SYMBOL_INLINE_TTL_SEC)
    except Exception as e:
        logger.warning(f"[Symbol Inline] failed to cache: {e}")

    return result
