# services/global_brief_service.py
"""
AI-generated global market brief for the public Finance World page.
Uses global news (general + crypto + forex) + optional market snapshot + LLM. Cached 30 min.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.ai.llm_service import get_llm_service
from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.finnhub_news_service import get_global_news_cached

logger = logging.getLogger(__name__)

CACHE_KEY_GLOBAL_BRIEF = "market:global_brief:v2"
TTL_GLOBAL_BRIEF_SEC = int(os.getenv("TTL_GLOBAL_BRIEF_SEC", "1800"))  # 30 min

SYSTEM_PROMPT = """You are a senior financial markets analyst writing a "World Brief" for the Finance Monitor dashboard. Your audience is investors and traders who need actionable intelligence.

Your task: Read the provided news headlines and snippets (and any market snapshot). Synthesize them into 4-6 short sections. For each section:
- headline: A clear, specific title (e.g. "Fed & Rates", "Geopolitical Risk", "Equities & Earnings", "Crypto & Risk Assets").
- cause: 2-3 sentences explaining what is happening and why it matters. Be specific: name entities, numbers, and catalysts from the news. Do not be generic.
- impact: One crisp "bottom line" sentence for markets or portfolios (e.g. "Expect volatility in short-duration bonds." or "Large-cap tech may see pressure on margins.").

Also output a single field: outlook — one sentence short-term market outlook or sentiment (e.g. "Cautiously risk-on with focus on rates and earnings.").

Rules:
- Ground every section in the provided news. Do not invent stories. If news is sparse, still produce 3-4 sections using the themes present and standard macro context (rates, inflation, growth, risk sentiment).
- Be concrete: mention sectors, regions, or asset classes where relevant.
- Tone: professional, direct, no fluff. No disclaimers inside the JSON.
- Output valid JSON only. No markdown, no code fences.

Schema (strict):
{
  "as_of": "<current ISO8601 UTC with Z suffix>",
  "market": "Global Markets",
  "outlook": "One sentence short-term outlook or sentiment.",
  "sections": [
    {
      "headline": "Title Case Section Title",
      "cause": "Specific 2-3 sentences from the news.",
      "impact": "One actionable bottom line."
    }
  ]
}"""


async def _get_merged_news_for_brief(max_items: int = 30) -> List[Dict[str, Any]]:
    """Fetch general, crypto, and forex news; merge, dedupe by url, sort by published_at desc."""
    general, crypto, forex = await asyncio.gather(
        get_global_news_cached(category="general", limit=15),
        get_global_news_cached(category="crypto", limit=8),
        get_global_news_cached(category="forex", limit=8),
    )
    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for item in (general or []) + (crypto or []) + (forex or []):
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(item)
    merged.sort(
        key=lambda x: x.get("published_at") or "",
        reverse=True,
    )
    return merged[:max_items]


def _build_user_prompt(
    news_items: List[Dict[str, Any]],
    market_snapshot: Optional[str] = None,
    max_items: int = 18,
) -> str:
    lines = ["## Recent headlines and snippets\n"]
    for item in (news_items or [])[:max_items]:
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        source = (item.get("source") or "").strip()
        pub = (item.get("published_at") or "").strip()
        if not title:
            continue
        block = f"- {title}"
        if snippet:
            block += f"\n  {snippet[:350]}{'…' if len(snippet) > 350 else ''}"
        if source or pub:
            block += f"\n  [{source}] {pub}"
        lines.append(block)
    if not any(line.startswith("- ") for line in lines):
        lines.append("\nNo specific headlines in feed. Use current macro themes: central bank policy, inflation, growth, earnings, geopolitics, and risk sentiment to write a brief World Brief.")
    if market_snapshot:
        lines.append("\n## Market snapshot (for context)\n" + market_snapshot)
    return "\n".join(lines)


async def get_global_brief_cached(
    *,
    force_refresh: bool = False,
    market_snapshot: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a cached or freshly generated global market brief.
    Shape: { as_of, market, sections }.
    If force_refresh=True, skip cache and call LLM.
    """
    if not force_refresh:
        cached = cache_get(CACHE_KEY_GLOBAL_BRIEF)
        if isinstance(cached, dict) and isinstance(cached.get("sections"), list):
            return cached

    try:
        news_items = await _get_merged_news_for_brief(max_items=30)
        user_prompt = _build_user_prompt(news_items, market_snapshot=market_snapshot, max_items=25)
        llm = get_llm_service()
        logger.info("global_brief_calling_llm force_refresh=%s news_count=%s", force_refresh, len(news_items))
        raw = await llm.generate_json(system=SYSTEM_PROMPT, user=user_prompt)
        if not isinstance(raw, dict):
            raise ValueError("LLM did not return a dict")

        sections = raw.get("sections") or []
        if not isinstance(sections, list):
            sections = []
        out_sections = []
        for s in sections:
            if not isinstance(s, dict):
                continue
            out_sections.append({
                "headline": (s.get("headline") or "").strip() or "Market update",
                "cause": (s.get("cause") or "").strip(),
                "impact": (s.get("impact") or "").strip(),
            })
        as_of = raw.get("as_of") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        market = (raw.get("market") or "Global Markets").strip()
        outlook = (raw.get("outlook") or "").strip() or None

        result = {
            "as_of": as_of,
            "market": market,
            "sections": out_sections,
            "outlook": outlook,
        }
        cache_set(CACHE_KEY_GLOBAL_BRIEF, result, ttl_seconds=TTL_GLOBAL_BRIEF_SEC)
        logger.info("global_brief_generated sections=%s outlook=%s", len(out_sections), bool(outlook))
        return result
    except Exception as e:
        logger.exception("global_brief_generation_failed: %s", e)
        return {
            "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "market": "Global Markets",
            "outlook": None,
            "sections": [
                {
                    "headline": "Brief unavailable",
                    "cause": "We couldn't generate the latest brief. Please try again in a few minutes.",
                    "impact": "Use the Refresh button to retry.",
                }
            ],
        }


CACHE_KEY_PREDICTIONS = "market:predictions:v1"
TTL_PREDICTIONS_SEC = int(os.getenv("TTL_PREDICTIONS_SEC", "1800"))  # 30 min

PREDICTIONS_SYSTEM_PROMPT = """You are a senior financial markets analyst. Your task is to output a single short-term market outlook or sentiment in one sentence.

Based on the provided news headlines and snippets, write exactly one sentence that captures the current market sentiment or short-term outlook (e.g. "Cautiously risk-on with focus on rates and earnings." or "Risk-off sentiment amid inflation concerns.").

Rules:
- Ground your sentence in the provided news; do not invent.
- Be direct and professional. No disclaimers.
- Output valid JSON only. No markdown, no code fences.

Schema (strict):
{
  "as_of": "<current ISO8601 UTC with Z suffix>",
  "outlook": "One sentence short-term market outlook or sentiment."
}"""


async def _generate_predictions_from_llm() -> Dict[str, Any]:
    """Call LLM with merged news to produce a single outlook sentence. Used for the dedicated Predictions card."""
    try:
        news_items = await _get_merged_news_for_brief(max_items=20)
        user_prompt = _build_user_prompt(news_items, market_snapshot=None, max_items=12)
        llm = get_llm_service()
        logger.info("predictions_calling_llm news_count=%s", len(news_items))
        raw = await llm.generate_json(system=PREDICTIONS_SYSTEM_PROMPT, user=user_prompt)
        if not isinstance(raw, dict):
            raise ValueError("LLM did not return a dict")
        outlook = (raw.get("outlook") or "").strip() or None
        as_of = (raw.get("as_of") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        return {"outlook": outlook, "as_of": as_of}
    except Exception as e:
        logger.exception("predictions_llm_failed: %s", e)
        return {
            "outlook": None,
            "as_of": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


async def get_predictions_cached(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Return cached or freshly generated predictions (outlook + as_of).
    Uses a dedicated short LLM call so Predictions can be refreshed independently from the full global brief.
    """
    if not force_refresh:
        cached = cache_get(CACHE_KEY_PREDICTIONS)
        if isinstance(cached, dict) and cached.get("as_of"):
            return cached

    result = await _generate_predictions_from_llm()
    if result.get("outlook"):
        cache_set(CACHE_KEY_PREDICTIONS, result, ttl_seconds=TTL_PREDICTIONS_SEC)
    return result
