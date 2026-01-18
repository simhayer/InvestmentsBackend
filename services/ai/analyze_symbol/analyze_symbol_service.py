# services/ai/analyze_symbol_service.py
import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from services.openai.client import llm
from services.openai.client import llm_mini
from services.vector.vector_store_service import VectorStoreService
from utils.common_helpers import timed

try:
    from services.tavily.client import search as tavily_search, compact_results as compact_tavily
except Exception:
    tavily_search = None  # type: ignore
    compact_tavily = None  # type: ignore

from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.cache.cache_backend import cache_set
from services.ai.technicals_pack import build_technical_pack, compact_tech_pack
from services.finnhub.finnhub_news_service import get_company_news_cached, shrink_news_items
from services.finnhub.finnhub_calender_service import get_earnings_calendar_compact_cached
from services.ai.helpers.analyze_symbol_helpers import (
    compute_market_snapshot, fetch_history_points, normalize_symbol,
    ck_task,
    ck_report,
    format_sec_chunks,
    get_fundamentals_with_cache,
    get_news_with_optional_tavily_fallback,
    get_sec_routed_context,
    json_dumps,
    trim_quote,
    trim_profile,
    trim_market_snapshot,
    trim_earnings_calendar,
    trim_sec_chunks,
    trim_news_items,
    shorten_text,
    build_technicals_text_from_pack,
    validate_report
    )

from .types import AnalysisReport, AgentState

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))  # 10m
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))              # 15m
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))   # 1h
TTL_ANALYSIS_REPORT_SEC = int(os.getenv("TTL_ANALYSIS_REPORT_SEC", "1800"))  # 30m per symbol


def _compute_data_quality(
    finnhub_data: Dict[str, Any],
    finnhub_gaps: List[str],
    sec_context: str,
    news_count: int
) -> List[str]:
    notes: List[str] = []

    if finnhub_gaps:
        notes.append(
            f"Fundamentals gaps present: {', '.join(finnhub_gaps[:10])}{'...' if len(finnhub_gaps) > 10 else ''}"
        )

    if not sec_context or len(sec_context.strip()) < 80:
        notes.append("SEC filing context is limited or unavailable.")

    if news_count < 3:
        notes.append("News coverage is light; catalysts may be incomplete or less reliable.")

    normalized = (finnhub_data or {}).get("normalized") or {}
    common_keys = [
        "market_cap",
        "pe_ttm",
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "debt_to_equity",
    ]
    missing_common = [k for k in common_keys if normalized.get(k) in (None, "", "NA")]
    if len(missing_common) >= 4:
        notes.append("Several key normalized metrics are missing; treat valuation/profitability conclusions cautiously.")

    return notes


# ----------------------------
# Graph Nodes
# ----------------------------
async def research_node(state: AgentState):
    symbol = normalize_symbol(state.get("symbol") or "")
    task_id = state.get("task_id", "no_task")

    # 1) Fundamentals (cached)
    finnhub_data, finnhub_gaps = await get_fundamentals_with_cache(
        symbol=symbol,
        state=state,
        ttl_seconds=TTL_FUNDAMENTALS_SEC,
        fetch_fundamentals_cached=fetch_fundamentals_cached,
    )

    normalized = (finnhub_data or {}).get("normalized") or {}

    # 2) Earnings Calendar (unchanged params)
    with timed("finnhub_earnings_calendar", logger, state=state):
        earnings_calendar = await get_earnings_calendar_compact_cached(
            symbol=symbol,
            window_days=120,
            limit=6,
            international=False,
        )
        earnings_calendar = sorted(earnings_calendar, key=lambda x: x.get("date") or "9999-99-99")

    # 3) News (Finnhub cached) + optional Tavily fallback (query unchanged)
    news_items, raw_str, used_tavily = await get_news_with_optional_tavily_fallback(
        symbol=symbol,
        state=state,
        ttl_tavily_seconds=TTL_TAVILY_SEC,
        get_company_news_cached=get_company_news_cached,
        tavily_search=tavily_search,
        compact_tavily=compact_tavily,
    )

    # 4) SEC Vector Context (Targeted Routing) (same section names/limits)
    sec_context, sec_business, sec_risks, sec_mda, sec_debug = get_sec_routed_context(
        symbol=symbol,
        state=state,
        task_id=task_id,
        vs=VectorStoreService(),
    )

    # 5) Final Assembly
    market_snapshot = compute_market_snapshot(finnhub_data)

    debug = {
        "symbol": symbol,
        "finnhub": {"gaps": finnhub_gaps, "normalized_preview": normalized},
        "news": {"count": len(news_items), "used_tavily": used_tavily},
        "sec": {"total_chunks": sec_debug.get("count", 0), "routed": True},
        "market_snapshot": market_snapshot,
        "earnings_calendar": {"preview": earnings_calendar[:2] if earnings_calendar else []},
    }

    return {
        "symbol": symbol,
        "raw_data": raw_str,
        "news_items": news_items,
        "finnhub_data": finnhub_data,
        "finnhub_gaps": finnhub_gaps,
        "sec_context": sec_context,
        "sec_business": sec_business,
        "sec_risks": sec_risks,
        "sec_mda": sec_mda,
        "market_snapshot": market_snapshot,
        "earnings_calendar": earnings_calendar,
        "debug": debug,
    }


async def drafts_node(state: AgentState):
    """
    Runs fundamentals/technicals/risks drafts in parallel.
    Uses targeted SEC sections for higher fidelity.
    """
    symbol = state.get("symbol") or ""

    sec_bus = state.get("sec_business", [])
    sec_risks_raw = state.get("sec_risks", [])
    sec_mda = state.get("sec_mda", [])

    finnhub_data = state.get("finnhub_data", {}) or {}
    market_snapshot = state.get("market_snapshot", {}) or {}
    normalized = finnhub_data.get("normalized", {}) or {}

    news_items = state.get("news_items", []) or []
    news_json = json.dumps(shrink_news_items(news_items, 10, 240), ensure_ascii=False)

    # unchanged history fetch params
    price_points, bench_points = await asyncio.gather(
        fetch_history_points(symbol, "2y", "1d"),
        fetch_history_points("SPY", "2y", "1d"),
    )

    tech_pack = compact_tech_pack(build_technical_pack(
        symbol=symbol,
        points=price_points,
        benchmark_symbol="SPY",
        benchmark_points=bench_points,
    ))

    async def do_fundamentals():
        prompt = f"""
            Analyze {symbol}. Write 3–6 Key Insights as short bullets.
            - At least one insight must identify a potential mispricing or market assumption.

            BUSINESS MODEL CONTEXT (SEC Item 1):
            {format_sec_chunks(sec_bus)}

            MANAGEMENT ANALYSIS (SEC Item 7):
            {format_sec_chunks(sec_mda)}

            FINNHUB NORMALIZED: {normalized}
            MARKET SNAPSHOT: {market_snapshot}
            NEWS: {news_json}

            Rules:
            - Ground at least 2 bullets in the SEC context (business model/drivers).
            - Use concrete metrics from FINNHUB NORMALIZED.
            Return plain text bullets.
            """
        with timed("llm_fundamentals",logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)},):
            res = await llm.ainvoke(prompt)
        return res.content or ""
    

    async def do_risks():
        prompt = f"""
            Identify stock risks for {symbol}. Be skeptical.

            OFFICIAL RISK FACTORS (SEC Item 1A):
            {format_sec_chunks(sec_risks_raw)}

            FINNHUB METRICS: {normalized}
            MARKET SNAPSHOT: {market_snapshot}

            Rules:
            - Prioritize threats from Item 1A (e.g., dependencies, regulatory, liquidity).
            - Specify HOW a risk hits revenue, margins, or valuation.
            - Avoid generic macro fluff.
            Return 4-8 short bullets.
            """
        with timed("llm_risks",logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)},):
            res = await llm_mini.ainvoke(prompt)
        return res.content or ""

    
    technicals_text = build_technicals_text_from_pack(symbol, tech_pack)

    fundamentals, risks = await asyncio.gather(
        do_fundamentals(), do_risks()
    )

    return {"fundamentals": fundamentals, "technicals": technicals_text, "risks": risks}


async def analyst_structured_node(state: AgentState):
    symbol = (state.get("symbol") or "").strip().upper()

    structured_llm = (
        llm.with_structured_output(AnalysisReport, method="function_calling")
        .bind(temperature=0.2)
    )

    finnhub_data = state.get("finnhub_data", {}) or {}
    normalized = (finnhub_data.get("normalized") or {})

    # ---- TRIM EVERYTHING ----
    profile_small = trim_profile(finnhub_data.get("profile") or {})
    quote_small = trim_quote(finnhub_data.get("quote") or {})
    market_snapshot_small = trim_market_snapshot(state.get("market_snapshot", {}) or {})

    earnings_calendar = state.get("earnings_calendar", []) or []
    # sort ascending so "next earnings" is truly next
    earnings_calendar = sorted(earnings_calendar, key=lambda x: x.get("date") or "9999-99-99")
    earnings_small = trim_earnings_calendar(earnings_calendar, max_items=2)
    next_date = earnings_small[0].get("date") if earnings_small else "Unknown"

    sec_bus_small = trim_sec_chunks(state.get("sec_business", []) or [], max_chunks=2, max_chars_each=650)
    sec_risks_small = trim_sec_chunks(state.get("sec_risks", []) or [], max_chunks=2, max_chars_each=650)
    sec_mda_small = trim_sec_chunks(state.get("sec_mda", []) or [], max_chunks=1, max_chars_each=650)

    news_items = state.get("news_items", []) or []
    news_small = trim_news_items(news_items, max_items=5, max_chars_each=220)

    finnhub_gaps = state.get("finnhub_gaps", []) or []

    # drafts are already “compressed thinking”
    fundamentals_draft = shorten_text(state.get("fundamentals", "") or "", 1200)
    risks_draft = shorten_text(state.get("risks", "") or "", 1200)
    technicals_text = shorten_text(state.get("technicals", "") or "", 500)

    critique = (state.get("critique") or "").strip()
    critique_block = f"\nCRITIQUE FIXES:\n{shorten_text(critique, 900)}\n" if critique and critique.upper() != "CLEAR" else ""

    data_quality_notes = _compute_data_quality(
        finnhub_data=finnhub_data,
        finnhub_gaps=finnhub_gaps,
        sec_context=state.get("sec_context", "") or "",
        news_count=len(news_items),
    )

    prompt = f"""
Return a JSON object that matches the AnalysisReport schema EXACTLY.

Definitive truth: next earnings date is {next_date}.

Hard rules:
- recommendation: Buy / Hold / Sell
- confidence: 0.0 to 1.0
- is_priced_in: true/false
- Use FINNHUB NORMALIZED as the only source of exact metrics.
- Scenarios must be exactly: Base, Bull, Bear.
- upcoming_catalysts: exactly 3 items.
  - If a catalyst is earnings-related, window MUST be exactly "{next_date}" (YYYY-MM-DD).

Brevity caps (speed):
- key_insights: 3–6
- stock_overflow_risks: 4–8
- thesis_points: <= 5
- key_debates: <= 4
- what_to_watch_next: <= 6
- evidence: optional; if present, note must be non-empty and <= 8 words.

Decision rule:
- The recommendation must explicitly reference at least one upside driver and one downside risk.

INPUTS (trimmed):
SYMBOL: {symbol}

NORMALIZED:
{json_dumps(normalized)}

PROFILE (trimmed):
{json_dumps(profile_small)}

QUOTE (trimmed):
{json_dumps(quote_small)}

MARKET SNAPSHOT (trimmed):
{json_dumps(market_snapshot_small)}

EARNINGS (trimmed):
{json_dumps(earnings_small)}

SEC BUSINESS SNIPS:
{json_dumps(sec_bus_small)}

SEC RISK SNIPS:
{json_dumps(sec_risks_small)}

SEC MD&A SNIPS:
{json_dumps(sec_mda_small)}

NEWS (trimmed):
{json_dumps(news_small)}

DRAFT INSIGHTS (seed):
{fundamentals_draft}

DRAFT RISKS (seed):
{risks_draft}

MARKET PERFORMANCE (deterministic):
{technicals_text}

DATA QUALITY NOTES (copy into report.data_quality_notes):
{json_dumps(data_quality_notes)}
{critique_block}
"""

    with timed("llm_analyst_structured_merged", logger, state=state, tags={"prompt_kb": round(len(prompt)/1024, 1)}):
        report = await structured_llm.ainvoke(prompt)

    iters = int(state.get("iterations", 0)) + 1
    return {"report": report.model_dump(), "iterations": iters}

async def critic_node(state: AgentState):
    report_obj = state.get("report", {}) or {}

    earnings_calendar = state.get("earnings_calendar", []) or []
    earnings_calendar = sorted(earnings_calendar, key=lambda x: x.get("date") or "9999-99-99")
    raw_next = earnings_calendar[0].get("date") if earnings_calendar else None
    next_date = str(raw_next) if raw_next else "Unknown"

    finnhub_gaps = state.get("finnhub_gaps", []) or []

    issues = validate_report(report_obj, next_earnings_date=next_date, finnhub_gaps=finnhub_gaps)
    if not issues:
        return {"is_valid": True, "critique": "CLEAR"}

    # Only send what matters
    small_payload = {
        "issues": issues,
        "recommendation": report_obj.get("recommendation"),
        "confidence": report_obj.get("confidence"),
        "upcoming_catalysts": report_obj.get("upcoming_catalysts"),
        "scenarios": report_obj.get("scenarios"),
    }

    prompt = f"""
You are a strict reviewer. Fix the report by addressing ONLY the listed issues.
Return 3–7 specific fix instructions. Do not add new sections.

ISSUES:
{json.dumps(issues, ensure_ascii=False)}

PAYLOAD:
{json.dumps(small_payload, ensure_ascii=False)}

If everything is fixed already, respond exactly: CLEAR
"""

    with timed("llm_critic_mini", logger, state=state, tags={"prompt_kb": round(len(prompt)/1024, 1)}):
        res = await llm_mini.ainvoke(prompt)  # your mini model

    txt = (res.content or "").strip()
    return {"is_valid": txt.upper() == "CLEAR", "critique": "" if txt.upper() == "CLEAR" else txt}


# ----------------------------
# Build Graph
# ----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("drafts", drafts_node)
workflow.add_node("analyst", analyst_structured_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "drafts")
workflow.add_edge("drafts", "analyst")
workflow.add_edge("analyst", "critic")

workflow.add_conditional_edges(
    "critic",
    lambda x: "end" if x.get("is_valid") else "revise",
    {"revise": "analyst", "end": END},
)

app_graph = workflow.compile()


# ----------------------------
# Task runner (Redis-backed)
# ----------------------------
async def run_analysis_task(symbol: str, task_id: str):
    """
    Stores task status + final structured report in Redis.
    API contract:
    { status, data: { report, total_seconds }, debug }
    """
    t0_total = time.perf_counter()
    symbol = normalize_symbol(symbol)
    task_key = ck_task(task_id)

    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_TASK_RESULT_SEC)

    try:
        final_state = await app_graph.ainvoke({"symbol": symbol, "iterations": 0, "task_id": task_id})
        total_dt = time.perf_counter() - t0_total

        report_obj = final_state.get("report")

        payload = {
            "status": "complete",
            "data": {"report": report_obj, "total_seconds": round(total_dt, 3)},
            "debug": final_state.get("debug", {}),
        }

        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)

        # Optional symbol-level cache
        try:
            cache_set(ck_report(symbol), report_obj, ttl_seconds=TTL_ANALYSIS_REPORT_SEC)
        except Exception:
            pass

        logger.info("[%s] %s: total=%.2fs", task_id, symbol, total_dt)

    except Exception as e:
        total_dt = time.perf_counter() - t0_total
        logger.exception("Analysis failed for %s (%s).", symbol, task_id)

        payload = {"status": "failed", "data": {"error": str(e), "total_seconds": round(total_dt, 3)}}
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
