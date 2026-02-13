# analyze_crypto_service.py
"""
AI analysis service for cryptocurrency assets.

Separate from the stock analysis service because:
- Crypto has no fundamentals (P/E, margins, earnings, analyst targets)
- Prompts focus on price action, risk metrics, market position, and sentiment
- Response schema is tailored to crypto-relevant fields
"""
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

# Cache TTL — crypto is more volatile, shorter TTL than stocks
CRYPTO_ANALYSIS_TTL_SEC = 6 * 3600   # 6 hours
CRYPTO_INLINE_TTL_SEC = 3 * 3600     # 3 hours


def _analysis_cache_key(symbol: str) -> str:
    return f"analysis:crypto:full:{symbol.upper()}"


def _inline_cache_key(symbol: str) -> str:
    return f"analysis:crypto:inline:{symbol.upper()}"


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
class CryptoAnalysisReport:
    symbol: str
    summary: str
    verdict: Verdict
    confidence: Confidence
    market_position: Dict[str, str]       # assessment + reasoning
    risk_profile: Dict[str, str]          # assessment + reasoning
    price_action: Dict[str, str]          # trend + reasoning
    bull_case: List[str]
    bear_case: List[str]
    risks: List[str]
    catalysts: List[str]
    technical_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "summary": self.summary,
            "verdict": self.verdict.value,
            "confidence": self.confidence.value,
            "marketPosition": self.market_position,
            "riskProfile": self.risk_profile,
            "priceAction": self.price_action,
            "bullCase": self.bull_case,
            "bearCase": self.bear_case,
            "risks": self.risks,
            "catalysts": self.catalysts,
            "technicalNotes": self.technical_notes,
        }


@dataclass
class CryptoInlineInsights:
    market_cap_badge: str
    volatility_callout: str
    trend_signal: str
    risk_flag: Optional[str]
    momentum_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marketCapBadge": self.market_cap_badge,
            "volatilityCallout": self.volatility_callout,
            "trendSignal": self.trend_signal,
            "riskFlag": self.risk_flag,
            "momentumNote": self.momentum_note,
        }


# ============================================================================
# PROMPTS — tailored for crypto (no earnings, no P/E, no analyst targets)
# ============================================================================

CRYPTO_SYSTEM_PROMPT = """You are a professional crypto asset analyst. Your role is to provide clear,
actionable analysis of cryptocurrency assets based on available market data.

Key principles:
- Be direct and opinionated — take a clear stance
- Support claims with the specific numbers provided (price, market cap, volatility, drawdown, Sharpe, etc.)
- Crypto has NO fundamentals like P/E, margins, or earnings — never invent these
- Focus on: market position, price action, risk/reward profile, and macro sentiment
- Acknowledge data limitations honestly — free data means no on-chain metrics
- Be realistic about crypto volatility and risk
- Use plain language, avoid hype

You always respond with valid JSON matching the requested schema.
Return ONLY JSON. No markdown. No extra text.
"""

CRYPTO_FULL_REPORT_PROMPT = """Analyze the following cryptocurrency data and provide a comprehensive investment analysis.

{context}

Respond with a JSON object matching this exact schema:
{{
    "summary": "2-3 sentence investment thesis for this crypto asset",
    "verdict": "Bullish" | "Bearish" | "Neutral",
    "confidence": "High" | "Medium" | "Low",
    "marketPosition": {{
        "assessment": "Leader" | "Mid-Cap" | "Small-Cap" | "Micro-Cap",
        "reasoning": "1-2 sentences about where this asset sits in the crypto landscape"
    }},
    "riskProfile": {{
        "assessment": "Conservative" | "Moderate" | "Aggressive" | "Speculative",
        "reasoning": "1-2 sentences citing volatility, drawdown, and Sharpe ratio"
    }},
    "priceAction": {{
        "trend": "Uptrend" | "Downtrend" | "Sideways" | "Recovery",
        "reasoning": "1-2 sentences about current price relative to 52-week range and moving averages"
    }},
    "bullCase": ["point 1", "point 2", "point 3"],
    "bearCase": ["point 1", "point 2", "point 3"],
    "risks": ["specific risk 1", "specific risk 2", "specific risk 3"],
    "catalysts": ["catalyst 1", "catalyst 2"],
    "technicalNotes": "1-2 sentences on price position vs 52-week range, MAs, golden/death cross, or null if no data"
}}

Critical instructions:
- Cite actual numbers from the data (market cap, volatility %, drawdown %, Sharpe, distance from high)
- Do NOT invent P/E ratios, earnings, or analyst targets — these don't exist for crypto
- If data is limited, lower your confidence to "Low" and state what's missing
- Be specific about risks (regulatory, liquidity, concentration, smart contract, etc.)
"""

CRYPTO_INLINE_PROMPT = """Based on the following crypto data, generate brief inline insights for UI display.
Each insight should be punchy and under 50 characters.

{context}

Respond with a JSON object:
{{
    "marketCapBadge": "e.g., 'Top 5 by market cap' or '$3M micro-cap'",
    "volatilityCallout": "e.g., '85% annualized vol — extreme risk'",
    "trendSignal": "e.g., '32% below 52W high, in downtrend'",
    "riskFlag": "Notable risk if any, or null",
    "momentumNote": "e.g., 'Above 50-day MA, golden cross forming'"
}}
"""


# ============================================================================
# SERVICE
# ============================================================================

class AICryptoAnalysisService:
    """LLM-powered crypto analysis. Provider/model selected by env config."""

    def __init__(self):
        self.llm = get_llm_service()

    def _parse_json(self, text: str) -> Dict[str, Any]:
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
        raw_obj = await self.llm.generate_json(
            system=CRYPTO_SYSTEM_PROMPT, user=prompt
        )
        if isinstance(raw_obj, dict):
            return raw_obj
        return self._parse_json(str(raw_obj))

    async def generate_full_report(
        self, context: str, symbol: str
    ) -> CryptoAnalysisReport:
        data = await self._ask(CRYPTO_FULL_REPORT_PROMPT.format(context=context))

        verdict = self._safe_enum(Verdict, data.get("verdict", "Neutral"), Verdict.NEUTRAL)
        conf = self._safe_enum(Confidence, data.get("confidence", "Medium"), Confidence.MEDIUM)

        return CryptoAnalysisReport(
            symbol=symbol,
            summary=data.get("summary", ""),
            verdict=verdict,
            confidence=conf,
            market_position=data.get("marketPosition", {}) or {},
            risk_profile=data.get("riskProfile", {}) or {},
            price_action=data.get("priceAction", {}) or {},
            bull_case=data.get("bullCase", []) or [],
            bear_case=data.get("bearCase", []) or [],
            risks=data.get("risks", []) or [],
            catalysts=data.get("catalysts", []) or [],
            technical_notes=data.get("technicalNotes"),
        )

    async def generate_inline_insights(self, context: str) -> CryptoInlineInsights:
        data = await self._ask(CRYPTO_INLINE_PROMPT.format(context=context))
        return CryptoInlineInsights(
            market_cap_badge=data.get("marketCapBadge", "") or "",
            volatility_callout=data.get("volatilityCallout", "") or "",
            trend_signal=data.get("trendSignal", "") or "",
            risk_flag=data.get("riskFlag"),
            momentum_note=data.get("momentumNote", "") or "",
        )


# ============================================================================
# CONVENIENCE FUNCTIONS (with Redis caching)
# ============================================================================

async def analyze_crypto(
    symbol: str,
    *,
    include_inline: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Full crypto analysis with Redis caching (6h TTL).
    """
    sym = symbol.upper()
    cache_key = _analysis_cache_key(sym)

    # Check cache
    if not force_refresh:
        cached = cache_get(cache_key)
        if cached and isinstance(cached, dict):
            logger.info(f"[Crypto Analysis] cache hit for {sym}")
            cached["cached"] = True
            return cached

    # Compute fresh analysis
    from services.ai.analyze_crypto.analyze_crypto_aggregator import aggregate_crypto_data

    bundle = await aggregate_crypto_data(sym)
    context = bundle.to_ai_context()

    ai = AICryptoAnalysisService()
    report = await ai.generate_full_report(context, sym)

    result: Dict[str, Any] = {
        "symbol": sym,
        "report": report.to_dict(),
        "riskMetrics": bundle.risk_metrics if bundle.risk_metrics else None,
        "marketData": {
            "currentPrice": bundle.current_price,
            "dayChangePct": bundle.day_change_pct,
            "marketCap": bundle.market_cap,
            "volume24h": bundle.volume_24h,
            "high52w": bundle.high_52w,
            "low52w": bundle.low_52w,
        },
        "dataGaps": bundle.gaps,
    }

    if include_inline:
        inline = await ai.generate_inline_insights(context)
        result["inline"] = inline.to_dict()

    result["lastAnalyzedAt"] = datetime.now(timezone.utc).isoformat()

    # Persist to Redis
    try:
        cache_set(cache_key, result, ttl_seconds=CRYPTO_ANALYSIS_TTL_SEC)
        logger.info(f"[Crypto Analysis] cached result for {sym}")
    except Exception as e:
        logger.warning(f"[Crypto Analysis] failed to cache: {e}")

    result["cached"] = False
    return result


async def get_crypto_insights(
    symbol: str,
    *,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """
    Inline insights with Redis caching (3h TTL).
    """
    sym = symbol.upper()
    cache_key = _inline_cache_key(sym)

    if not force_refresh:
        cached = cache_get(cache_key)
        if cached and isinstance(cached, dict):
            logger.info(f"[Crypto Inline] cache hit for {sym}")
            return cached

    from services.ai.analyze_crypto.analyze_crypto_aggregator import aggregate_crypto_data

    bundle = await aggregate_crypto_data(sym, include_news=False)
    context = bundle.to_ai_context()

    ai = AICryptoAnalysisService()
    inline = await ai.generate_inline_insights(context)
    result = inline.to_dict()

    try:
        cache_set(cache_key, result, ttl_seconds=CRYPTO_INLINE_TTL_SEC)
    except Exception as e:
        logger.warning(f"[Crypto Inline] failed to cache: {e}")

    return result
