# services/ai/analyze_symbol/analyze_symbol_service.py
import os
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Histogram, Counter

from services.openai.client import llm
from services.vector.vector_store_service import VectorStoreService
from services.cache.cache_backend import cache_set, cache_get
from utils.common_helpers import timed

from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.finnhub.peer_benchmark_service import (
    fetch_peer_benchmark_cached,
    build_peer_summary,
    build_peer_comparison_ready,
)
from services.finnhub.finnhub_news_service import get_company_news_cached
from services.finnhub.finnhub_calender_service import get_earnings_calendar_compact_cached
from services.ai.technicals_pack import build_technical_pack, compact_tech_pack

from services.ai.helpers.analyze_symbol_helpers import (
    compute_market_snapshot,
    fetch_history_points,
    normalize_symbol,
    ck_task,
    ck_report,
    get_fundamentals_with_cache,
    get_news_with_optional_tavily_fallback,
    get_sec_routed_context,
    json_dumps,
    compute_data_quality,
    build_technicals_text_from_pack,
)

from .types import AgentState, CoreAnalysis, NewsBrief, MaterialDriver, RiskResearchOutput
from .facts_pack_service import (
    build_facts_pack,
    build_current_performance,
    build_price_outlook,
)

from services.tavily.client import search as tavily_search, compact_results as compact_tavily

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Cache TTLs
TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))
TTL_ANALYSIS_REPORT_SEC = int(os.getenv("TTL_ANALYSIS_REPORT_SEC", "1800"))
TTL_CHECKPOINT_SEC = int(os.getenv("TTL_CHECKPOINT_SEC", "3600"))

# Data limits
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "12"))
MAX_SEC_EXCERPTS = int(os.getenv("MAX_SEC_EXCERPTS", "10"))
MAX_INSIGHTS = 3
MAX_THESIS_POINTS = 3
MAX_CATALYSTS = 3
MAX_RISKS = 5
MAX_OPPORTUNITIES = 5

# Concurrency controls
FINNHUB_SEM = asyncio.Semaphore(int(os.getenv("FINNHUB_CONCURRENCY", "5")))
SEC_SEM = asyncio.Semaphore(int(os.getenv("SEC_CONCURRENCY", "2")))
TAVILY_SEM = asyncio.Semaphore(int(os.getenv("TAVILY_CONCURRENCY", "2")))

# Shared instances
VS = VectorStoreService()

# Metrics
RESEARCH_DURATION = Histogram(
    'research_node_duration_seconds',
    'Time spent in research node',
    ['symbol']
)
ANALYSIS_FAILURES = Counter(
    'analysis_failures_total',
    'Total analysis failures',
    ['symbol', 'stage']
)
ANALYSIS_SUCCESS = Counter(
    'analysis_success_total',
    'Total successful analyses',
    ['symbol']
)

class NewsRelevance(str, Enum):
    """News relevance levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataSource(str, Enum):
    """Data source types for tracking"""
    FINNHUB = "finnhub"
    SEC = "sec"
    TAVILY = "tavily"
    CACHE = "cache"

def compact_for_llm(
    items: List[Dict[str, Any]], 
    field_map: Dict[str, Tuple[List[str], int]], 
    limit: int
) -> List[Dict[str, Any]]:
    items = (items or [])[:limit]
    out: List[Dict[str, Any]] = []
    
    for item in items:
        compact = {}
        for output_field, (input_aliases, max_len) in field_map.items():
            for alias in input_aliases:
                if val := item.get(alias):
                    compact[output_field] = str(val).strip()[:max_len]
                    break
            # Set None if no value found
            if output_field not in compact:
                compact[output_field] = None
        out.append(compact)
    
    return out


def compact_news_for_llm(news_items: List[Dict[str, Any]], limit: int = MAX_NEWS_ITEMS) -> List[Dict[str, Any]]:
    field_map = {
        "headline": (["headline", "title"], 160),
        "summary": (["summary", "snippet"], 320),
        "source": (["source"], 60),
        "datetime": (["datetime", "published_at", "time", "published"], None),
        "url": (["url"], None),
    }
    return compact_for_llm(news_items, field_map, limit)


def compact_sec_risks_for_llm(sec_risks: List[Dict[str, Any]], limit: int = MAX_SEC_EXCERPTS) -> List[Dict[str, Any]]:
    field_map = {
        "text": (["text", "chunk", "content"], 520),
        "section": (["section", "label", "route"], 40),
        "source": (["source", "doc", "filing"], 80),
    }
    return compact_for_llm(sec_risks, field_map, limit)


def safe_sorted_earnings(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items or [], key=lambda x: x.get("date") or "9999-99-99")


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_key_risks_for_ui(risks: List[MaterialDriver]) -> List[str]:
    out: List[str] = []
    for risk in risks[:MAX_RISKS]:
        evidence = risk.evidence[0] if risk.evidence else ""
        line = f"{risk.label}: {risk.mechanism}"
        if evidence:
            line += f" (Evidence: {evidence})"
        out.append(line)
    return out


def extract_news_relevance(state: AgentState) -> NewsRelevance:
    brief = state.get("news_brief") or {}
    val = (brief.get("news_relevance") or "low")
    
    if not isinstance(val, str):
        return NewsRelevance.LOW
    
    val = val.lower().strip()
    try:
        return NewsRelevance(val)
    except ValueError:
        return NewsRelevance.LOW


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(task_id: str, stage: str, data: Dict[str, Any]) -> None:
    """Save intermediate results for debugging and resume capability."""
    checkpoint_key = f"checkpoint:{task_id}:{stage}"
    try:
        cache_set(checkpoint_key, data, ttl_seconds=TTL_CHECKPOINT_SEC)
        logger.debug(f"Checkpoint saved: {checkpoint_key}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint {checkpoint_key}: {e}")


def load_checkpoint(task_id: str, stage: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint if available."""
    checkpoint_key = f"checkpoint:{task_id}:{stage}"
    try:
        return cache_get(checkpoint_key)
    except Exception:
        return None


# ============================================================================
# DATA FETCHING WITH RETRY & ERROR HANDLING
# ============================================================================

class DataFetchError(Exception):
    """Custom exception for data fetching failures."""
    pass


async def sem_run(sem: asyncio.Semaphore, coro):
    """Run coroutine with semaphore limiting."""
    async with sem:
        return await coro


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(DataFetchError)
)
async def fetch_fundamentals_with_retry(symbol: str, state: AgentState) -> Tuple[Dict[str, Any], List[str]]:
    """Fetch fundamentals with automatic retry."""
    try:
        return await get_fundamentals_with_cache(
            symbol=symbol,
            state=state,
            ttl_seconds=TTL_FUNDAMENTALS_SEC,
            fetch_fundamentals_cached=fetch_fundamentals_cached,
        )
    except Exception as e:
        logger.error(f"Fundamentals fetch failed for {symbol}: {e}")
        raise DataFetchError(f"Failed to fetch fundamentals: {e}")


async def build_technicals(symbol: str) -> str:
    """Build technical analysis text."""
    try:
        price_points, bench_points = await asyncio.gather(
            fetch_history_points(symbol, "2y", "1d"),
            fetch_history_points("SPY", "2y", "1d"),
        )
        
        tech_pack = compact_tech_pack(
            build_technical_pack(
                symbol=symbol,
                points=price_points,
                benchmark_symbol="SPY",
                benchmark_points=bench_points,
            )
        )
        
        return build_technicals_text_from_pack(symbol, tech_pack)
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return f"Technical analysis unavailable: {str(e)}"


# ============================================================================
# VALIDATION
# ============================================================================

def validate_core_analysis(core: Dict[str, Any]) -> List[str]:
    """
    Validate core analysis output for quality standards.
    Returns list of quality issues found.
    """
    issues = []
    
    # Check completeness
    if len(core.get("key_insights", [])) != MAX_INSIGHTS:
        issues.append(f"Expected {MAX_INSIGHTS} key insights, got {len(core.get('key_insights', []))}")
    
    if len(core.get("thesis_points", [])) != MAX_THESIS_POINTS:
        issues.append(f"Expected {MAX_THESIS_POINTS} thesis points, got {len(core.get('thesis_points', []))}")
    
    if len(core.get("upcoming_catalysts", [])) != MAX_CATALYSTS:
        issues.append(f"Expected {MAX_CATALYSTS} catalysts, got {len(core.get('upcoming_catalysts', []))}")
    
    # Check evidence quality - insights should have numbers
    for insight in core.get("key_insights", []):
        evidence = insight.get("evidence", "") if isinstance(insight, dict) else ""
        if not any(char.isdigit() for char in str(evidence)):
            issues.append(f"Insight lacks numeric evidence: {insight.get('label', 'unknown')}")
    
    # Check for duplicate catalysts
    catalysts = [c.get("event", "") if isinstance(c, dict) else "" 
                 for c in core.get("upcoming_catalysts", [])]
    if len(catalysts) != len(set(catalysts)):
        issues.append("Duplicate catalysts detected")
    
    return issues


# ============================================================================
# GRAPH NODES
# ============================================================================

async def research_node(state: AgentState) -> Dict[str, Any]:
    """
    Parallel data gathering from all sources.
    Uses graceful degradation - partial failures don't kill entire task.
    """
    symbol = normalize_symbol(state.get("symbol") or "")
    task_id = state.get("task_id", "no_task")
    
    # Check for cached checkpoint
    if checkpoint := load_checkpoint(task_id, "research"):
        logger.info(f"Loaded research checkpoint for {task_id}")
        return checkpoint
    
    with RESEARCH_DURATION.labels(symbol=symbol).time():
        
        async def get_fund():
            with timed("finnhub_fundamentals", logger, state=state):
                try:
                    return await fetch_fundamentals_with_retry(symbol, state)
                except Exception as e:
                    logger.error(f"Fundamentals failed for {symbol}: {e}")
                    return {}, ["fundamentals_unavailable"]
        
        async def get_earnings():
            with timed("finnhub_earnings_calendar", logger, state=state):
                try:
                    items = await sem_run(
                        FINNHUB_SEM,
                        get_earnings_calendar_compact_cached(
                            symbol=symbol,
                            window_days=120,
                            limit=6,
                            international=False,
                        ),
                    )
                    return safe_sorted_earnings(items)
                except Exception as e:
                    logger.error(f"Earnings calendar failed for {symbol}: {e}")
                    return []
        
        async def get_peers():
            with timed("finnhub_peer_benchmark", logger, state=state):
                try:
                    res = await sem_run(
                        FINNHUB_SEM, 
                        fetch_peer_benchmark_cached(symbol, timeout_s=5.0)
                    )
                    return (res.data or {}), (res.gaps or [])
                except Exception as e:
                    logger.error(f"Peer benchmark failed for {symbol}: {e}")
                    return {}, ["peer_data_unavailable"]
        
        async def get_news():
            with timed("finnhub_news", logger, state=state):
                try:
                    return await get_news_with_optional_tavily_fallback(
                        symbol=symbol,
                        state=state,
                        ttl_tavily_seconds=TTL_TAVILY_SEC,
                        get_company_news_cached=get_company_news_cached,
                        tavily_search=tavily_search,
                        compact_tavily=compact_tavily,
                    )
                except Exception as e:
                    logger.error(f"News fetch failed for {symbol}: {e}")
                    return [], "", False
        
        async def get_sec():
            with timed("sec_vector_routing", logger, state=state):
                try:
                    return await sem_run(
                        SEC_SEM,
                        asyncio.to_thread(
                            get_sec_routed_context,
                            symbol=symbol,
                            state=state,
                            task_id=task_id,
                            vs=VS,
                        ),
                    )
                except Exception as e:
                    logger.error(f"SEC data failed for {symbol}: {e}")
                    return "", "", [], "", {"count": 0, "error": str(e)}
        
        async def get_tech():
            with timed("technicals_pack", logger, state=state):
                return await build_technicals(symbol)
        
        # Gather all data in parallel - use return_exceptions for graceful degradation
        results = await asyncio.gather(
            get_fund(),
            get_earnings(),
            get_peers(),
            get_news(),
            get_sec(),
            get_tech(),
            return_exceptions=True
        )
        
        # Unpack results with error handling
        finnhub_data, finnhub_gaps = results[0] if not isinstance(results[0], Exception) else ({}, ["fundamentals_error"])
        earnings_calendar = results[1] if not isinstance(results[1], Exception) else []
        peer_benchmark, peer_gaps = results[2] if not isinstance(results[2], Exception) else ({}, ["peer_error"])
        news_items, raw_str, used_tavily = results[3] if not isinstance(results[3], Exception) else ([], "", False)
        sec_context, sec_business, sec_risks, sec_mda, sec_debug = results[4] if not isinstance(results[4], Exception) else ("", "", [], "", {"count": 0})
        technicals_text = results[5] if not isinstance(results[5], Exception) else "Technical analysis unavailable"
    
    # Build derived data
    market_snapshot = compute_market_snapshot(finnhub_data)
    peer_ready = build_peer_comparison_ready(peer_benchmark)
    
    # Track data completeness
    data_completeness = {
        "fundamentals": not isinstance(results[0], Exception) and bool(finnhub_data),
        "earnings": not isinstance(results[1], Exception) and bool(earnings_calendar),
        "peers": not isinstance(results[2], Exception) and bool(peer_benchmark),
        "news": not isinstance(results[3], Exception) and bool(news_items),
        "sec": not isinstance(results[4], Exception) and bool(sec_context),
        "technicals": not isinstance(results[5], Exception),
    }
    
    debug = {
        "symbol": symbol,
        "data_completeness": data_completeness,
        "news": {
            "count": len(news_items), 
            "used_tavily": used_tavily,
            "source": DataSource.TAVILY.value if used_tavily else DataSource.FINNHUB.value
        },
        "sec": {
            "total_chunks": sec_debug.get("count", 0), 
            "routed": True,
            "error": sec_debug.get("error")
        },
        "earnings_calendar": {
            "count": len(earnings_calendar),
            "preview": earnings_calendar[:2] if earnings_calendar else []
        },
        "peers": {
            "count": len(peer_ready.get("peers_used") or []), 
            "gaps": peer_gaps
        },
    }
    
    result = {
        "symbol": symbol,
        "task_id": task_id,
        "raw_data": raw_str,
        "news_items": news_items,
        "finnhub_data": finnhub_data,
        "finnhub_gaps": finnhub_gaps,
        "earnings_calendar": earnings_calendar,
        "peer_benchmark": peer_benchmark,
        "peer_gaps": peer_gaps,
        "peer_comparison_ready": peer_ready,
        "sec_context": sec_context,
        "sec_business": sec_business,
        "sec_risks": sec_risks,
        "sec_mda": sec_mda,
        "market_snapshot": market_snapshot,
        "technicals": technicals_text,
        "debug": debug,
        "data_completeness": data_completeness,
    }
    
    # Save checkpoint
    save_checkpoint(task_id, "research", result)
    
    return result


async def facts_pack_node(state: AgentState) -> Dict[str, Any]:
    """Build structured facts pack from raw research data."""
    facts_pack = build_facts_pack(
        market_snapshot=state.get("market_snapshot") or {},
        quote=(state.get("finnhub_data", {}).get("quote") or {}),
        peer_ready=(state.get("peer_comparison_ready") or {}),
        earnings_small=safe_sorted_earnings(state.get("earnings_calendar", []))[:1],
        data_quality_notes=compute_data_quality(
            finnhub_data=state.get("finnhub_data", {}),
            finnhub_gaps=state.get("finnhub_gaps", []),
            sec_context=state.get("sec_context", ""),
            news_count=len(state.get("news_items", [])),
        ),
    )
    
    return {"facts_pack": facts_pack.model_dump()}


async def news_brief_node(state: AgentState) -> Dict[str, Any]:
    symbol = state["symbol"]
    news_items = state.get("news_items", [])
    
    # Early exit if no news
    if not news_items:
        logger.info(f"No news available for {symbol}, skipping news brief")
        return {
            "news_brief": {
                "news_relevance": NewsRelevance.LOW.value,
                "what_changed_today": [],
                "key_themes": [],
                "catalyst_candidates": [],
                "risk_signals": [],
            }
        }
    
    llm_nb = llm.with_structured_output(NewsBrief).bind(temperature=0.2)
    news_compact = compact_news_for_llm(news_items, limit=MAX_NEWS_ITEMS)
    
    prompt = f"""
You are generating a DAILY NEWS BRIEF for {symbol}.

STRICT RULES:
- Only use the provided NEWS items below
- Do not guess or infer beyond what's explicitly stated
- If news is generic or not company-specific, set news_relevance="low"
- Every bullet MUST include evidence from a headline/summary
- Avoid macro market recaps unless clearly tied to {symbol}

OUTPUT REQUIREMENTS:
- what_changed_today: 2-4 bullets (each with evidence)
- key_themes: 2-4 themes
- catalyst_candidates: 2-4 concrete catalysts (must be distinct, no duplicates)
- risk_signals: 0-4 bullets (only if strong evidence supports)
- news_relevance: "high" | "medium" | "low"

QUALITY STANDARDS:
- "high" relevance = company-specific developments with material impact
- "medium" relevance = relevant but incremental news
- "low" relevance = generic market news or no company-specific items

NEWS ITEMS ({len(news_compact)} total):
{json_dumps(news_compact)}
""".strip()
    
    try:
        brief = await llm_nb.ainvoke(prompt)
        
        debug_info = state.get("debug", {}) or {}
        debug_info["news_brief"] = {
            "relevance": getattr(brief, "news_relevance", None),
            "what_changed_count": len(getattr(brief, "what_changed_today", []) or []),
            "catalysts_count": len(getattr(brief, "catalyst_candidates", []) or []),
            "risk_signals_count": len(getattr(brief, "risk_signals", []) or []),
        }
        
        return {
            "news_brief": brief.model_dump(),
            "debug": debug_info
        }
        
    except Exception as e:
        logger.error(f"News brief generation failed for {symbol}: {e}")
        ANALYSIS_FAILURES.labels(symbol=symbol, stage="news_brief").inc()
        # Return minimal valid structure
        return {
            "news_brief": {
                "news_relevance": NewsRelevance.LOW.value,
                "what_changed_today": [],
                "key_themes": ["News analysis unavailable"],
                "catalyst_candidates": [],
                "risk_signals": [],
            }
        }


async def risk_research_node(state: AgentState) -> Dict[str, Any]:
    symbol = state["symbol"]
    llm_rr = llm.with_structured_output(RiskResearchOutput).bind(temperature=0.2)
    
    facts = state.get("facts_pack") or {}
    news_brief = state.get("news_brief") or {}
    relevance = extract_news_relevance(state)
    
    # Only include raw news when brief says it's not junk
    news_compact: List[Dict[str, Any]] = []
    if relevance in {NewsRelevance.HIGH, NewsRelevance.MEDIUM}:
        news_compact = compact_news_for_llm(state.get("news_items", []), limit=MAX_NEWS_ITEMS)
    
    sec_compact = compact_sec_risks_for_llm(state.get("sec_risks", []), limit=MAX_SEC_EXCERPTS)
    
    base_prompt = f"""
You are a senior buy-side equity analyst conducting materiality research for {symbol}.

DATA SOURCES PROVIDED:
1. FACTS PACK - Structured metrics, labels, and fundamental data
2. NEWS BRIEF - Curated "what changed today" + catalyst candidates
3. SEC RISK EXCERPTS - Risk factor snippets from latest filings
{"4. RAW NEWS - Additional evidence (use sparingly, news brief already curated key points)" if news_compact else ""}

YOUR MISSION:
Identify what is *materially relevant* for {symbol} RIGHT NOW.

CRITICAL RULES:
- Do NOT guess or speculate beyond provided evidence
- If evidence is weak, lower confidence or omit the item entirely
- If NEWS BRIEF news_relevance is "low", rely primarily on SEC RISK EXCERPTS + FACTS PACK
- Focus on company-specific factors, not generic macro themes
- Every risk/opportunity MUST be actionable and specific

OUTPUT STRUCTURE (STRICT):
- risks: 0-{MAX_RISKS} items (direction="risk")
  * Each needs: label, mechanism, evidence (1-2 snippets), confidence (0.0-1.0), watch_items (up to 3)
- opportunities: 0-{MAX_OPPORTUNITIES} items (direction="opportunity")
  * Same requirements as risks
- watch_list: 3-5 bullets for monitoring

MECHANISM QUALITY:
- Explain HOW it impacts: revenue, margins, valuation multiple, or risk premium
- Be specific: "Margin compression from rising input costs" not "Costs may increase"

EVIDENCE REQUIREMENTS:
- Must be traceable to NEWS BRIEF, RAW NEWS (if provided), or SEC excerpts
- Use direct quotes or clear paraphrases
- Cite source type in evidence

CONFIDENCE SCORING:
- 0.8-1.0: Strong, recent, company-specific evidence
- 0.5-0.7: Moderate evidence or somewhat speculative
- 0.0-0.4: Weak evidence, highly speculative (avoid these)

AVOID:
- Repeating the same idea across multiple items
- Generic risks that apply to any company ("market volatility")
- Opportunities without clear evidence

---

FACTS PACK:
{json_dumps(facts)}

NEWS BRIEF (relevance={relevance.value}):
{json_dumps(news_brief)}
""".strip()
    
    if news_compact:
        base_prompt += f"""

RAW NEWS (for additional evidence only - main themes already in NEWS BRIEF):
{json_dumps(news_compact)}
""".rstrip()
    
    base_prompt += f"""

SEC RISK EXCERPTS:
{json_dumps(sec_compact)}
""".rstrip()
    
    try:
        rr = await llm_rr.ainvoke(base_prompt)
        
        debug_info = state.get("debug", {}) or {}
        debug_info["materiality"] = {
            "risks_count": len(rr.risks or []),
            "opportunities_count": len(rr.opportunities or []),
            "watch_list_count": len(rr.watch_list or []),
            "news_relevance": relevance.value,
            "raw_news_included": bool(news_compact),
        }
        
        return {
            "risk_research": rr.model_dump(),
            "debug": debug_info,
        }
        
    except Exception as e:
        logger.error(f"Risk research failed for {symbol}: {e}")
        ANALYSIS_FAILURES.labels(symbol=symbol, stage="risk_research").inc()
        # Return minimal valid structure
        return {
            "risk_research": {
                "risks": [],
                "opportunities": [],
                "watch_list": ["Risk analysis unavailable"],
            }
        }


async def analyst_core_node(state: AgentState) -> Dict[str, Any]:
    symbol = state["symbol"]
    llm_core = llm.with_structured_output(CoreAnalysis).bind(temperature=0.2)
    
    facts = state["facts_pack"]
    risk_research = state.get("risk_research") or {}
    news_brief = state.get("news_brief") or {}
    relevance = extract_news_relevance(state)
    
    prompt = f"""
You are a senior equity analyst at a top-tier hedge fund analyzing {symbol}.

DATA PROVIDED:
- FACTS PACK: Structured metrics (treat as ground truth)
- MATERIALITY: Curated risks/opportunities with evidence
- NEWS BRIEF: Daily changes and catalyst candidates

YOUR MISSION:
Generate actionable investment analysis with clear evidence.

OUTPUT STRUCTURE (STRICT - NO EXCEPTIONS):

1. key_insights: EXACTLY {MAX_INSIGHTS} insights
   Requirements for EACH insight:
   - Must reference a SPECIFIC metric from FACTS PACK (P/E ratio, margin %, growth rate, price level, etc.)
   - Format: [Observation] + [Implication]
   - Evidence: One concrete sentence with numbers
   - Example: "Trading at P/E of 45x vs sector median 28x suggests premium valuation. Evidence: Current P/E 45.2x is 61% above peer group median."

2. thesis_points: EXACTLY {MAX_THESIS_POINTS} points
   Requirements for EACH point:
   - Format: [Claim] because [Evidence from facts/materiality]
   - Must connect to revenue/margin/growth/valuation drivers
   - Example: "Margin expansion likely to continue because gross margins improved 340bps YoY to 42.1% per latest filing, supported by operating leverage from scale."

3. upcoming_catalysts: EXACTLY {MAX_CATALYSTS} catalysts
   Requirements:
   - Must be DISTINCT event types (no duplicates)
   - Maximum ONE earnings-related catalyst
   - Other catalysts must be different categories: regulatory decision, product launch, M&A, partnership, FDA approval, etc.
   - Each catalyst needs: event, expected_timeframe, potential_impact
   - Prioritize by: impact × probability
   - If news_relevance is NOT "low", use NEWS BRIEF catalyst_candidates as starting point

4. scenarios: EXACTLY 3 scenarios
   Required names: "Base", "Bull", "Bear"
   Each scenario needs:
   - name: exactly as specified above
   - description: 2-3 sentences
   - key_drivers: 2-3 bullet points
   - probability: decimal (must sum to ~1.0 across all scenarios)

QUALITY STANDARDS:
- Every insight/thesis MUST include actual numbers (percentages, ratios, dollar amounts)
- Evidence MUST be traceable to FACTS PACK or MATERIALITY
- No generic market commentary without company specifics
- No catalog-style risk lists
- Scenarios should reflect actual data dispersion, not just +/- 20%

WHAT THE MARKET MAY BE MISSING:
- Identify ONE non-consensus view based on data
- Explain why consensus might be wrong
- Ground it in facts, not speculation

---

FACTS PACK:
{json_dumps(facts)}

NEWS BRIEF (relevance={relevance.value}):
{json_dumps(news_brief)}

MATERIALITY (risks + opportunities):
{json_dumps(risk_research)}

Generate your analysis now.
""".strip()
    
    try:
        core = await llm_core.ainvoke(prompt)
        
        # Validate output quality
        quality_issues = validate_core_analysis(core.model_dump())
        
        debug_info = state.get("debug", {}) or {}
        debug_info["analyst_core"] = {
            "insights_count": len(getattr(core, "key_insights", []) or []),
            "thesis_count": len(getattr(core, "thesis_points", []) or []),
            "catalysts_count": len(getattr(core, "upcoming_catalysts", []) or []),
            "quality_issues": quality_issues,
        }
        
        if quality_issues:
            logger.warning(f"Quality issues in core analysis for {symbol}: {quality_issues}")
        
        return {
            "core_analysis": core.model_dump(),
            "quality_warnings": quality_issues,
            "debug": debug_info,
        }
        
    except Exception as e:
        logger.error(f"Core analysis failed for {symbol}: {e}")
        ANALYSIS_FAILURES.labels(symbol=symbol, stage="analyst_core").inc()
        raise  # This is critical - let it fail


async def validate_node(state: AgentState) -> Dict[str, Any]:
    """
    Quality validation before final assembly.
    Ensures report meets standards.
    """
    core = state.get("core_analysis", {})
    quality_issues = validate_core_analysis(core)
    
    # Add to existing warnings
    existing_warnings = state.get("quality_warnings", [])
    all_warnings = existing_warnings + quality_issues
    
    if all_warnings:
        logger.warning(f"Quality validation found {len(all_warnings)} issues for {state.get('symbol')}")
    
    return {"quality_warnings": all_warnings}


def assemble_report_node(state: AgentState) -> Dict[str, Any]:
    symbol = state["symbol"]
    core = state["core_analysis"]
    facts = state["facts_pack"]
    peer_ready = state.get("peer_comparison_ready") or {}
    
    risk_research = state.get("risk_research") or {}
    rr_risks = risk_research.get("risks") or []
    rr_watch = risk_research.get("watch_list") or []
    
    news_brief = state.get("news_brief") or {}
    what_changed = news_brief.get("what_changed_today") or []
    
    # Convert structured risks to UI-friendly format
    key_risks_ui: List[str] = []
    try:
        parsed_risks = [MaterialDriver.model_validate(x) for x in rr_risks]
        key_risks_ui = format_key_risks_for_ui(parsed_risks)
    except Exception as e:
        logger.error(f"Failed to format risks for {symbol}: {e}")
        key_risks_ui = []
    
    # Build final report
    report = {
        # Core analysis (from LLM)
        **core,
        
        # Identity
        "symbol": symbol,
        
        # Deterministic sections (no LLM)
        "current_performance": build_current_performance(facts),
        "price_outlook": build_price_outlook(facts, core),
        
        # Materiality-driven sections
        "key_risks": key_risks_ui[:MAX_RISKS],
        "what_to_watch_next": (rr_watch or [])[:5],
        
        # News section
        "what_changed_today": {
            "bullets": [
                b.get("bullet") 
                for b in what_changed 
                if isinstance(b, dict) and b.get("bullet")
            ]
        },
        
        # Peer comparison
        "peer_comparison": peer_ready if peer_ready.get("peers_used") else None,
        "peer_comparison_summary": (
            build_peer_summary(peer_ready) 
            if peer_ready.get("peers_used") 
            else []
        ),
        
        # Quality & completeness
        "data_quality_notes": facts.get("data_quality_notes", []),
        "data_completeness": state.get("data_completeness", {}),
        "quality_warnings": state.get("quality_warnings", []),
    }
    
    return {"report": report}


# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("research", research_node)
workflow.add_node("facts_pack", facts_pack_node)
workflow.add_node("news_brief", news_brief_node)
workflow.add_node("risk_research", risk_research_node)
workflow.add_node("analyst_core", analyst_core_node)
workflow.add_node("validate", validate_node)
workflow.add_node("assemble", assemble_report_node)

# Define strict linear flow
workflow.set_entry_point("research")
workflow.add_edge("research", "facts_pack")
workflow.add_edge("facts_pack", "news_brief")
workflow.add_edge("news_brief", "risk_research")
workflow.add_edge("risk_research", "analyst_core")
workflow.add_edge("analyst_core", "validate")
workflow.add_edge("validate", "assemble")
workflow.add_edge("assemble", END)

# Compile graph
app_graph = workflow.compile()


# ============================================================================
# TASK RUNNER (Redis-backed)
# ============================================================================

async def run_analysis_task(symbol: str, task_id: str) -> None:
    """
    Execute full analysis workflow and store results in cache.
    
    Args:
        symbol: Stock ticker symbol
        task_id: Unique task identifier for tracking
    """
    t0_total = time.perf_counter()
    symbol = normalize_symbol(symbol)
    task_key = ck_task(task_id)
    
    # Set initial status
    cache_set(
        task_key, 
        {"status": "processing", "data": None}, 
        ttl_seconds=TTL_TASK_RESULT_SEC
    )
    
    try:
        # Run the full graph
        final_state = await app_graph.ainvoke({
            "symbol": symbol,
            "iterations": 0,
            "task_id": task_id
        })
        
        total_duration = time.perf_counter() - t0_total
        report_obj = final_state.get("report")
        
        # Build success payload
        payload = {
            "status": "complete",
            "data": {
                "report": report_obj,
                "total_seconds": round(total_duration, 3),
            },
            "debug": final_state.get("debug", {}),
            "quality_warnings": final_state.get("quality_warnings", []),
            "data_completeness": final_state.get("data_completeness", {}),
        }
        
        # Store task result
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
        
        # Also cache the report separately for quick access
        try:
            cache_set(
                ck_report(symbol), 
                report_obj, 
                ttl_seconds=TTL_ANALYSIS_REPORT_SEC
            )
        except Exception as e:
            logger.warning(f"Failed to cache report for {symbol}: {e}")
        
        # Metrics
        ANALYSIS_SUCCESS.labels(symbol=symbol).inc()
        logger.info(
            f"[{task_id}] {symbol}: Analysis complete in {total_duration:.2f}s "
            f"(warnings={len(final_state.get('quality_warnings', []))})"
        )
        
    except Exception as e:
        total_duration = time.perf_counter() - t0_total
        logger.exception(f"Analysis failed for {symbol} ({task_id})")
        
        # Store failure
        payload = {
            "status": "failed",
            "data": {
                "error": str(e),
                "error_type": type(e).__name__,
                "total_seconds": round(total_duration, 3),
            }
        }
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
        
        # Metrics
        ANALYSIS_FAILURES.labels(symbol=symbol, stage="graph_execution").inc()