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
            res = await llm.ainvoke(prompt)
        return res.content or ""

    async def do_technicals():
        prompt = f"""
            Write a "Market Performance" section for {symbol} using:
            {tech_pack}

            Rules:
            - Mention returns (1D, 1M, 1Y) and trend vs MAs.
            - Mention relative performance vs SPY.
            """
        with timed("llm_technicals", logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)},):
            res = await llm.ainvoke(prompt)
        return res.content or ""

    fundamentals, technicals, risks = await asyncio.gather(
        do_fundamentals(), do_technicals(), do_risks()
    )

    return {"fundamentals": fundamentals, "technicals": technicals, "risks": risks}


async def thesis_builder_node(state: AgentState):
    """
    Produces unified thesis + catalysts + scenarios + debates drafts.
    Enforces strict temporal grounding using earnings_calendar.
    """
    symbol = state.get("symbol") or ""

    sec_bus = state.get("sec_business", [])
    sec_risks_raw = state.get("sec_risks", [])
    sec_mda = state.get("sec_mda", [])

    earnings_calendar = state.get("earnings_calendar", []) or []
    earnings_json = json_dumps(earnings_calendar)
    next_date = earnings_calendar[0].get("date") if earnings_calendar else "Unknown"

    prompt = f"""
        You are writing a professional-grade investment note for {symbol}. Output STRICT JSON only.

        SEC BUSINESS (Item 1):
        {format_sec_chunks(sec_bus)}

        SEC RISK FACTORS (Item 1A):
        {format_sec_chunks(sec_risks_raw)}

        SEC MD&A (Item 7):
        {format_sec_chunks(sec_mda)}

        FINNHUB NORMALIZED: {state.get('finnhub_data', {}).get('normalized')}
        MARKET SNAPSHOT: {state.get('market_snapshot')}
        EARNINGS CALENDAR: {earnings_json}
        DRAFT INSIGHTS: {state.get('fundamentals','')}
        DRAFT RISKS: {state.get('risks','')}

        DEFINITIVE TRUTH: The next earnings date is {next_date}.

        Task: Create a JSON object with unified_thesis, thesis_points, upcoming_catalysts, scenarios, market_expectations, key_debates, what_to_watch_next.

        STRICT RULES:
        1. UPCOMING CATALYSTS: If you list an Earnings catalyst, the 'window' MUST be EXACTLY "{next_date}".
        2. Use YYYY-MM-DD format for all dated windows.
        3. At least 1 thesis_point MUST reference the SEC Risk Factors or Business Model.
        4. Scenarios must be exactly: Base, Bull, Bear.

        Return STRICT JSON only.
        """

    with timed("llm_thesis_builder",logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)},):
        res = await llm.ainvoke(prompt)

    txt = (res.content or "").strip()
    try:
        obj = json.loads(txt)
    except Exception:
        obj = {"unified_thesis": "Not available", "thesis_points": [], "upcoming_catalysts": []}

    return {
        "unified_thesis": obj.get("unified_thesis", "Not available"),
        "thesis_points_draft": obj.get("thesis_points", []),
        "catalysts_draft": obj.get("upcoming_catalysts", []),
        "scenarios_draft": obj.get("scenarios", []),
        "market_expectations_draft": obj.get("market_expectations", []),
        "debates_draft": obj.get("key_debates", []),
        "what_to_watch_next_draft": obj.get("what_to_watch_next", []),
    }

async def analyst_structured_node(state: AgentState):
    """
    Faster final report builder:
    - Shrinks prompt size aggressively
    - Uses drafts + minimal sources
    - Optional cache to skip LLM work on repeats
    """
    symbol = (state.get("symbol") or "").strip().upper()
    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling")

    # --- Inputs (trimmed) ---
    finnhub_data = state.get("finnhub_data", {}) or {}
    normalized = (finnhub_data.get("normalized") or {})
    profile = trim_profile(finnhub_data.get("profile") or {})
    quote = trim_quote(finnhub_data.get("quote") or {})

    market_snapshot = trim_market_snapshot(state.get("market_snapshot", {}) or {})
    earnings_calendar = trim_earnings_calendar(state.get("earnings_calendar", []) or [], max_items=3)

    # SEC chunks: keep small; if empty, don’t paste huge empty blocks
    sec_bus_small = trim_sec_chunks(state.get("sec_business", []) or [], max_chunks=2, max_chars_each=650)
    sec_risks_small = trim_sec_chunks(state.get("sec_risks", []) or [], max_chunks=2, max_chars_each=650)
    sec_mda_small = trim_sec_chunks(state.get("sec_mda", []) or [], max_chunks=1, max_chars_each=650)

    news_small = trim_news_items(state.get("news_items", []) or [], max_items=5, max_chars_each=220)

    finnhub_gaps = state.get("finnhub_gaps", []) or []
    critique = (state.get("critique") or "").strip()
    critique_block = f"\nCRITIQUE FIXES:\n{shorten_text(critique, 900)}\n" if critique and critique.upper() != "CLEAR" else ""

    # Drafts: these are already “thinking” outputs; keep them, but cap size.
    fundamentals_draft = shorten_text(state.get("fundamentals", "") or "", 1200)
    risks_draft = shorten_text(state.get("risks", "") or "", 1200)
    technicals_draft = shorten_text(state.get("technicals", "") or "", 900)

    thesis_seed = {
        "unified_thesis": shorten_text(state.get("unified_thesis", "Not available") or "", 550),
        "thesis_points": (state.get("thesis_points_draft", []) or [])[:6],
        "upcoming_catalysts": (state.get("catalysts_draft", []) or [])[:4],
        "scenarios": (state.get("scenarios_draft", []) or [])[:3],
        "market_expectations": (state.get("market_expectations_draft", []) or [])[:4],
        "key_debates": (state.get("debates_draft", []) or [])[:4],
        "what_to_watch_next": (state.get("what_to_watch_next_draft", []) or [])[:6],
    }

    # --- Data quality notes ---
    data_quality_notes = _compute_data_quality(
        finnhub_data=finnhub_data,
        finnhub_gaps=finnhub_gaps,
        sec_context=state.get("sec_context", "") or "",
        news_count=len(state.get("news_items", []) or []),
    )

    # --- Prompt (small + focused) ---
    prompt = f"""
Return a JSON object that matches the AnalysisReport schema EXACTLY.

Hard rules:
- recommendation: Buy / Hold / Sell
- confidence: 0.0 to 1.0
- is_priced_in: true/false
- Do NOT invent precise financial metrics (use NORMALIZED only).
- If a field is missing, use "Not available"/"Unknown" rather than guessing.

Grounding rules:
- Use SEC snippets ONLY for business model + risk factors.
- Use NEWS only as context; don’t treat it as authoritative.
- Keep language tight and non-fluffy.

Required structure:
- key_insights: must include >=1 SEC-grounded insight if SEC snippets exist; otherwise state SEC not available.
- stock_overflow_risks: must include >=2 SEC-grounded risks if SEC snippets exist; otherwise state SEC not available.
- upcoming_catalysts: >=3 items; each must include:
  trigger, mechanism, impact_channels, probability, priced_in, key_watch_items
- scenarios: exactly 3 items: Base, Bull, Bear

Confidence caps:
- If FINNHUB GAPS non-empty -> confidence <= 0.55
- Else if SEC snippets empty AND NEWS empty -> confidence <= 0.55
- Else confidence <= 0.75

INPUTS (trimmed):
SYMBOL: {symbol}

NORMALIZED (only source of exact metrics):
{json.dumps(normalized, ensure_ascii=False)}

PROFILE (trimmed):
{json.dumps(profile, ensure_ascii=False)}

QUOTE (trimmed):
{json.dumps(quote, ensure_ascii=False)}

MARKET SNAPSHOT (trimmed):
{json.dumps(market_snapshot, ensure_ascii=False)}

EARNINGS CALENDAR (trimmed):
{json.dumps(earnings_calendar, ensure_ascii=False)}

SEC BUSINESS SNIPPETS (small):
{json.dumps(sec_bus_small, ensure_ascii=False)}

SEC RISK SNIPPETS (small):
{json.dumps(sec_risks_small, ensure_ascii=False)}

SEC MD&A SNIPPETS (small):
{json.dumps(sec_mda_small, ensure_ascii=False)}

NEWS (trimmed):
{json.dumps(news_small, ensure_ascii=False)}

DRAFTS (use as starting point, tighten if needed):
- fundamentals_draft: {fundamentals_draft}
- technicals_draft: {technicals_draft}
- risks_draft: {risks_draft}

THESIS SEED (already drafted; keep structure):
{json.dumps(thesis_seed, ensure_ascii=False)}

DATA QUALITY NOTES (copy into report.data_quality_notes; add if needed):
{json.dumps(data_quality_notes, ensure_ascii=False)}
{critique_block}

Also:
- current_performance: short paragraph, no invented numbers
- price_outlook: must align with recommendation; explain base/bull/bear logic briefly
"""

    with timed(
        "llm_analyst_structured_v2",
        logger,
        state=state,
        tags={"prompt_kb": round(len(prompt) / 1024, 1)},
    ):
        report = await structured_llm.ainvoke(prompt)

    iters = int(state.get("iterations", 0)) + 1
    report_dict = report.model_dump()

    return {"report": report_dict, "iterations": iters}

async def critic_node(state: AgentState):
    """
    Cheap guardrail. Max 1 revision for speed.
    """
    iters = int(state.get("iterations", 0))
    report_obj = state.get("report", {}) or {}

    prompt = f"""
        You are a strict reviewer. Check this report for:
        - missing catalysts structure (trigger/mechanism/impact_channels/probability/priced_in/watch items)
        - generic/fluffy claims
        - contradictions between normalized metrics and conclusion
        - recommendation not matching reasoning
        - confidence too high/low given uncertainty
        - scenarios missing Base/Bull/Bear or too vague
        - data_quality_notes missing obvious gaps

        If it is good enough to ship, respond exactly: CLEAR
        Otherwise respond with 3–7 specific fixes.

        REPORT JSON:
        {json.dumps(report_obj, ensure_ascii=False)}
        """

    with timed("llm_critic",logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)},):
        res = await llm_mini.ainvoke(prompt)

    txt = (res.content or "").strip()
    is_clear = txt.upper() == "CLEAR"

    if iters >= 1:
        return {"is_valid": True, "critique": "" if is_clear else txt}

    return {"is_valid": bool(is_clear), "critique": "" if is_clear else txt}


# ----------------------------
# Build Graph
# ----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("drafts", drafts_node)
workflow.add_node("thesis_builder", thesis_builder_node)
workflow.add_node("analyst", analyst_structured_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "drafts")
workflow.add_edge("drafts", "thesis_builder")
workflow.add_edge("thesis_builder", "analyst")
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
