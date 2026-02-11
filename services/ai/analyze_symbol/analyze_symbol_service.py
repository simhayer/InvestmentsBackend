# ai_stock_analysis.py
from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Literal


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
# LLM CLIENT INTERFACE
# ============================================================================

class LLMClient(Protocol):
    async def generate_json(self, *, system: str, user: str) -> str:
        """Return raw text that should be JSON."""


# ============================================================================
# OPENAI CLIENT
# ============================================================================

class OpenAIClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]


# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

class AnthropicClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": 2000,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]


# ============================================================================
# GEMINI CLIENT (optionally with web search)
# ============================================================================

GeminiThinking = Literal["LOW", "MEDIUM", "HIGH"]

class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        use_web: bool = True,
        thinking_level: GeminiThinking = "HIGH",
        temperature: float = 0.3,
    ):
        self.api_key = api_key
        self.model = model
        self.use_web = use_web
        self.thinking_level = thinking_level
        self.temperature = temperature

    def _normalize_thinking(self) -> str:
        lvl = (self.thinking_level or "HIGH").upper()
        m = (self.model or "").lower()

        if "pro" in m:
            return lvl if lvl in ("LOW", "HIGH") else "HIGH"
        if "flash" in m:
            return lvl if lvl in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
        return lvl if lvl in ("LOW", "HIGH") else "HIGH"

    async def generate_json(self, *, system: str, user: str) -> str:
        return await asyncio.to_thread(self._sync_call, system, user)

    def _sync_call(self, system: str, user: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        combined = f"{system}\n\n{user}"
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=combined)])
        ]

        tools = None
        if self.use_web:
            tools = [types.Tool(googleSearch=types.GoogleSearch())]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=self._normalize_thinking()),
            tools=tools,
            temperature=self.temperature,
        )

        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = getattr(resp, "text", None)
        return text if text else str(resp)


# ============================================================================
# (OPTIONAL) "CLOUD" CLIENT STUB
# - For your own gateway / internal service later
# - Caller doesn't change; only this resolver changes
# ============================================================================

class CloudLLMClient:
    """
    Example: call your own cloud gateway: POST /v1/llm/analyze
    You can implement however you want later.
    """
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                f"{self.base_url}/v1/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"system": system, "user": user},
            )
            r.raise_for_status()
            data = r.json()
            # you decide your gateway shape; assume {"text": "..."}
            return data["text"]


# ============================================================================
# SERVICE (CALLER NEVER CHOOSES PROVIDER)
# ============================================================================

class AIAnalysisService:
    """
    Public API stays stable.
    Internally selects the provider/model based on env/config only.
    """

    def __init__(self):
        self.client = self._resolve_client()

    def _resolve_client(self) -> LLMClient:
        """
        Decide provider/model here only.
        Callers never care.
        """
        provider = (os.getenv("AI_PROVIDER") or "gemini").lower()

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
            return OpenAIClient(api_key=api_key, model=model)

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("Missing ANTHROPIC_API_KEY")
            model = os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
            return AnthropicClient(api_key=api_key, model=model)

        if provider == "cloud":
            base_url = os.getenv("CLOUD_LLM_BASE_URL", "")
            api_key = os.getenv("CLOUD_LLM_API_KEY", "")
            if not base_url or not api_key:
                raise ValueError("Missing CLOUD_LLM_BASE_URL or CLOUD_LLM_API_KEY")
            return CloudLLMClient(base_url=base_url, api_key=api_key)

        # default: gemini
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
        use_web = (os.getenv("GEMINI_USE_WEB", "1") == "1")
        thinking = (os.getenv("GEMINI_THINKING_LEVEL") or "HIGH").upper()
        return GeminiClient(
            api_key=api_key,
            model=model,
            use_web=use_web,
            thinking_level=thinking,  # auto-normalized inside client
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),
        )

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
        raw = await self.client.generate_json(system=SYSTEM_PROMPT, user=prompt)
        return self._parse_json(raw)

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

async def analyze_stock(symbol: str, *, include_inline: bool = True) -> Dict[str, Any]:
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data

    bundle = await aggregate_stock_data(symbol)
    context = bundle.to_ai_context()

    ai = AIAnalysisService()
    report = await ai.generate_full_report(context, symbol)

    result: Dict[str, Any] = {
        "symbol": symbol,
        "report": report.to_dict(),
        "dataGaps": bundle.gaps,
        "rawData": bundle.to_dict(),
    }

    if include_inline:
        inline = await ai.generate_inline_insights(context)
        result["inline"] = inline.to_dict()

    return result


async def get_stock_insights(symbol: str) -> Dict[str, str]:
    from services.ai.analyze_symbol.analyze_symbol_aggregator import aggregate_stock_data

    bundle = await aggregate_stock_data(symbol, include_news=False)
    context = bundle.to_ai_context()

    ai = AIAnalysisService()
    inline = await ai.generate_inline_insights(context)
    return inline.to_dict()
