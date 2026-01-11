# services/ai/analyze_symbol_service.py
import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from services.openai.client import llm
from database import get_db
from services.vector.vector_store_service import VectorStoreService
from utils.common_helpers import timed, safe_float

try:
    from services.tavily.client import search as tavily_search, compact_results as compact_tavily
except Exception:
    tavily_search = None  # type: ignore
    compact_tavily = None  # type: ignore

from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.cache.cache_backend import cache_get, cache_set
from services.ai.technicals_pack import build_technical_pack, compact_tech_pack
from services.finnhub.finnhub_news_service import get_company_news_cached, shrink_news_items
from services.finnhub.finnhub_calender_service import get_earnings_calendar_compact_cached
from services.ai.helpers.analyze_symbol_helpers import compute_market_snapshot, fetch_history_points

from .types import AnalysisReport, AgentState 

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ----------------------------
# Helpers / Cache Keys
# ----------------------------
def _preview(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))  # 10m
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))              # 15m
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))   # 1h
TTL_ANALYSIS_REPORT_SEC = int(os.getenv("TTL_ANALYSIS_REPORT_SEC", "1800"))  # 30m per symbol (optional)

def _ck_fund(symbol: str) -> str:
    return f"ANALYZE:FUND:{(symbol or '').strip().upper()}"

def _ck_tav(symbol: str) -> str:
    return f"ANALYZE:TAV:{(symbol or '').strip().upper()}"

def _ck_task(task_id: str) -> str:
    return f"ANALYZE:TASK:{(task_id or '').strip()}"

def _ck_report(symbol: str) -> str:
    return f"ANALYZE:REPORT:{(symbol or '').strip().upper()}"

def _compute_data_quality(
    finnhub_data: Dict[str, Any],
    finnhub_gaps: List[str],
    sec_context: str,
    news_count: int
) -> List[str]:
    notes: List[str] = []

    if finnhub_gaps:
        notes.append(f"Fundamentals gaps present: {', '.join(finnhub_gaps[:10])}{'...' if len(finnhub_gaps) > 10 else ''}")

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
    symbol = (state.get("symbol") or "").strip().upper()
    task_id = state.get("task_id", "no_task")

    # 1) Fundamentals (cached)
    fin_key = _ck_fund(symbol)
    cached_f = cache_get(fin_key)
    if isinstance(cached_f, dict) and "data" in cached_f and "gaps" in cached_f:
        finnhub_data = cached_f.get("data") or {}
        finnhub_gaps = cached_f.get("gaps") or []
    else:
        with timed("finnhub_fundamentals", state, logger):
            finres = await fetch_fundamentals_cached(symbol, timeout_s=5.0)
        finnhub_data = finres.data
        finnhub_gaps = finres.gaps
        cache_set(fin_key, {"data": finnhub_data, "gaps": finnhub_gaps}, ttl_seconds=TTL_FUNDAMENTALS_SEC)

    normalized = (finnhub_data or {}).get("normalized") or {}
    profile = (finnhub_data or {}).get("profile") or {}
    quote = (finnhub_data or {}).get("quote") or {}

    # 2) Earnings Calendar
    with timed("finnhub_earnings_calendar", state, logger):
        earnings_calendar = await get_earnings_calendar_compact_cached(
            symbol=symbol,
            window_days=120,
            limit=6,
            international=False,
        )

    # 3) News (Finnhub cached) + optional Tavily fallback
    news_items: List[Dict[str, Any]] = []
    news_compact = ""
    used_tavily = False

    try:
        with timed("finnhub_news", state, logger):
            news_payload = await get_company_news_cached(symbol, days_back=10, limit=10)
        news_items = news_payload.get("items") or []
        news_compact = news_payload.get("compact") or ""
    except Exception:
        news_items = []
        news_compact = ""

    raw_str = news_compact

    if (len(news_items) < 3) and tavily_search is not None and compact_tavily is not None:
        tav_key = _ck_tav(symbol)
        cached_t = cache_get(tav_key)
        if isinstance(cached_t, dict) and isinstance(cached_t.get("raw"), str):
            raw_str = cached_t["raw"]
            used_tavily = True
        else:
            try:
                with timed("tavily_search", state, logger):
                    results = await tavily_search(
                        query=f"latest news, upcoming catalysts, bull and bear case for {symbol}",
                        max_results=8,
                        search_depth="advanced",
                    )
                raw_str = compact_tavily(results)
                cache_set(tav_key, {"raw": raw_str}, ttl_seconds=TTL_TAVILY_SEC)
                used_tavily = True
            except Exception:
                raw_str = news_compact

    # 4) SEC Vector Context (Targeted Routing)
    sec_business, sec_risks, sec_mda = [], [], []
    sec_debug = {"count": 0, "chunks": []}
    
    db_gen = get_db()
    try:
        db = next(db_gen)
        vs = VectorStoreService()

        # Run targeted section searches in parallel
        with timed("sec_vector_routing", state, logger):
            # VectorStoreService.get_context_for_analysis may be synchronous; run in threads to parallelize
            sec_business = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 1", limit=5)
            sec_risks   = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 1A", limit=8)
            sec_mda     = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 7", limit=5)

            print(f"[{task_id}] SEC routing for {symbol}: ", 'sec_business:', len(sec_business), 'sec_risks:', len(sec_risks), 'sec_mda:', len(sec_mda))

        # Combine for a general fallback context
        all_chunks = sec_business + sec_risks + sec_mda
        sec_debug["count"] = len(all_chunks)
        sec_debug["chunks"] = [{"score": c.get("score"), "section": c.get("metadata", {}).get("section_name")} for c in all_chunks[:10]]
        
        sec_context = "\n".join([f"- {c.get('content')}" for c in all_chunks])[:12000]

    except Exception as e:
        logger.warning(f"[{task_id}] SEC routing failed for {symbol}: {e}")
        sec_context = ""
    finally:
        next(db_gen, None)

    # 5) Final Assembly
    market_snapshot = compute_market_snapshot(finnhub_data)

    debug = {
        "symbol": symbol,
        "finnhub": {"gaps": finnhub_gaps, "normalized_preview": normalized},
        "news": {"count": len(news_items), "used_tavily": used_tavily},
        "sec": {"total_chunks": sec_debug["count"], "routed": True},
        "market_snapshot": market_snapshot,
        "earnings_calendar": {"preview": earnings_calendar[:2] if earnings_calendar else []}
    }

    return {
        "symbol": symbol,
        "raw_data": raw_str,
        "news_items": news_items,
        "finnhub_data": finnhub_data,
        "finnhub_gaps": finnhub_gaps,
        "sec_context": sec_context,
        "sec_business": sec_business,  # Used by Fundamentals draft
        "sec_risks": sec_risks,        # Used by Risks draft
        "sec_mda": sec_mda,            # Used by Fundamentals/Thesis
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

    # Targeted SEC buckets from research_node
    sec_bus = state.get("sec_business", [])
    sec_risks_raw = state.get("sec_risks", [])
    sec_mda = state.get("sec_mda", [])
    
    # Existing data
    finnhub_data = state.get("finnhub_data", {}) or {}
    market_snapshot = state.get("market_snapshot", {}) or {}
    normalized = finnhub_data.get("normalized", {}) or {}
    profile = finnhub_data.get("profile", {}) or {}
    quote = finnhub_data.get("quote", {}) or {}

    news_items = state.get("news_items", []) or []
    news_json = json.dumps(shrink_news_items(news_items, 10, 240), ensure_ascii=False)

    # Helper to clean up chunk lists for prompts
    def format_sec(chunks):
        if not chunks: return "No specific SEC section context found."
        return "\n".join([f"- {c['content']}" for c in chunks])

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
        with timed("llm_fundamentals", state, logger):
            prompt = f"""
Analyze {symbol}. Write 3–6 Key Insights as short bullets.

BUSINESS MODEL CONTEXT (SEC Item 1):
{format_sec(sec_bus)}

MANAGEMENT ANALYSIS (SEC Item 7):
{format_sec(sec_mda)}

FINNHUB NORMALIZED: {normalized}
MARKET SNAPSHOT: {market_snapshot}
NEWS: {news_json}

Rules:
- Ground at least 2 bullets in the SEC context (business model/drivers).
- Use concrete metrics from FINNHUB NORMALIZED.
Return plain text bullets.
"""
            res = await llm.ainvoke(prompt)
        return res.content or ""

    async def do_risks():
        with timed("llm_risks", state, logger):
            prompt = f"""
Identify stock risks for {symbol}. Be skeptical.

OFFICIAL RISK FACTORS (SEC Item 1A):
{format_sec(sec_risks_raw)}

FINNHUB METRICS: {normalized}
MARKET SNAPSHOT: {market_snapshot}

Rules:
- Prioritize threats from Item 1A (e.g., dependencies, regulatory, liquidity).
- Specify HOW a risk hits revenue, margins, or valuation.
- Avoid generic macro fluff.
Return 4-8 short bullets.
"""
            res = await llm.ainvoke(prompt)
        return res.content or ""

    async def do_technicals():
        with timed("llm_technicals", state, logger):
            prompt = f"""
Write a "Market Performance" section for {symbol} using:
{tech_pack}

Rules:
- Mention returns (1D, 1M, 1Y) and trend vs MAs.
- Mention relative performance vs SPY.
"""
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
    
    # Grouped context for the 'Big Picture' view
    sec_context = f"""
    BUSINESS (Item 1): {state.get('sec_business', [])[:3]}
    RISKS (Item 1A): {state.get('sec_risks', [])[:3]}
    MANAGEMENT (Item 7): {state.get('sec_mda', [])[:3]}
    """

    earnings_calendar = state.get("earnings_calendar", []) or []
    earnings_json = json.dumps(earnings_calendar, ensure_ascii=False)
    
    # Get definitive next date for prompt reinforcement
    next_date = earnings_calendar[0].get("date") if earnings_calendar else "Unknown"

    with timed("llm_thesis_builder", state, logger):
        prompt = f"""
You are writing a professional-grade investment note for {symbol}. Output STRICT JSON only.

SEC SECTION CONTEXT:
{sec_context}

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
        res = await llm.ainvoke(prompt)

    txt = (res.content or "").strip()
    try:
        obj = json.loads(txt)
    except Exception:
        obj = {"unified_thesis": "Not available", "thesis_points": [], "upcoming_catalysts": []} # Fallback

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
    Produces final AnalysisReport (structured output).
    Applies critique if needed.
    """
    symbol = state.get("symbol") or ""
    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling")

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE / FIXES TO APPLY:\n{critique}\n" if critique and critique.upper() != "CLEAR" else ""

    finnhub_data = state.get("finnhub_data", {}) or {}
    finnhub_gaps = state.get("finnhub_gaps", []) or []
    raw_data = state.get("raw_data", "") or ""
    sec_context = state.get("sec_context", "") or ""
    market_snapshot = state.get("market_snapshot", {}) or {}

    normalized = (finnhub_data or {}).get("normalized") or {}
    profile = (finnhub_data or {}).get("profile") or {}
    quote = (finnhub_data or {}).get("quote") or {}

    # drafts from earlier nodes
    unified_thesis = state.get("unified_thesis", "Not available")
    thesis_points = state.get("thesis_points_draft", [])
    catalysts = state.get("catalysts_draft", [])
    scenarios = state.get("scenarios_draft", [])
    market_expectations = state.get("market_expectations_draft", [])
    key_debates = state.get("debates_draft", [])
    what_to_watch_next = state.get("what_to_watch_next_draft", [])

    earnings_calendar = state.get("earnings_calendar", []) or []

    news_items = state.get("news_items", []) or []
    data_quality_notes = _compute_data_quality(
        finnhub_data=finnhub_data,
        finnhub_gaps=finnhub_gaps,
        sec_context=sec_context,
        news_count=len(news_items),
    )

    with timed("llm_analyst_structured", state, logger):
        prompt = f"""
Return a JSON object that matches the AnalysisReport schema EXACTLY.

Core rules:
- recommendation: Buy / Hold / Sell
- confidence: 0.0 to 1.0
- is_priced_in: true/false
- DO NOT output exact "fair value" / price target numbers unless they appear in FINNHUB NORMALIZED.
- If something is missing, say "Not available" / "Unknown" rather than guessing.

Source rules (STRICT):
- FINNHUB NORMALIZED is the only source for precise financial metrics/ratios.
- SEC context is authoritative for business model, dependencies, margin drivers, legal/regulatory, liquidity narrative, risk factors.
- WEB/NEWS is contextual only and must be treated as non-authoritative.

Grounding requirements:
- key_insights must include >=1 insight grounded in SEC context (or explicitly say SEC not available).
- stock_overflow_risks must include >=2 risks grounded in SEC context (or explicitly say SEC not available).
- upcoming_catalysts must have >=3 items and each must include trigger + mechanism + impact_channels + probability + priced_in + key_watch_items.
- scenarios must include exactly 3 items: Base, Bull, Bear.

Confidence guidance:
- If FINNHUB has gaps -> cap confidence at 0.55
- Else if BOTH SEC context and WEB/NEWS context are empty -> cap confidence at 0.55
- Else allow up to 0.75, keep it modest if conclusions rely mostly on WEB/NEWS.

Inputs:
SYMBOL: {symbol}

SEC CONTEXT:
{sec_context}

FINNHUB NORMALIZED:
{normalized}

PROFILE:
{profile}

QUOTE:
{quote}

MARKET SNAPSHOT:
{market_snapshot}

FINNHUB GAPS:
{finnhub_gaps}

EARNINGS CALENDAR:
{json.dumps(earnings_calendar, ensure_ascii=False)}

WEB/NEWS CONTEXT:
{raw_data}

DRAFT INSIGHTS:
{state.get("fundamentals","")}

DRAFT PERFORMANCE:
{state.get("technicals","")}

DRAFT RISKS:
{state.get("risks","")}

THESIS BUILDER DRAFTS (use these; improve if needed without breaking structure):
- unified_thesis: {unified_thesis}
- thesis_points: {json.dumps(thesis_points, ensure_ascii=False)}
- upcoming_catalysts: {json.dumps(catalysts, ensure_ascii=False)}
- scenarios: {json.dumps(scenarios, ensure_ascii=False)}
- market_expectations: {json.dumps(market_expectations, ensure_ascii=False)}
- key_debates: {json.dumps(key_debates, ensure_ascii=False)}
- what_to_watch_next: {json.dumps(what_to_watch_next, ensure_ascii=False)}

DATA QUALITY NOTES (copy into report.data_quality_notes; add more if needed):
{json.dumps(data_quality_notes, ensure_ascii=False)}

Also:
- current_performance should be a short paragraph (use drafts, no invented numbers).
- price_outlook must include base/bull/bear logic and align with recommendation.
{critique_block}
"""
        report = await structured_llm.ainvoke(prompt)

    iters = int(state.get("iterations", 0)) + 1
    return {"report": report.model_dump(), "iterations": iters}


async def critic_node(state: AgentState):
    """
    Cheap guardrail. Max 1 revision for speed.
    """
    iters = int(state.get("iterations", 0))
    report_obj = state.get("report", {}) or {}

    with timed("llm_critic", state, logger):
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
        res = await llm.ainvoke(prompt)

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
    symbol = (symbol or "").strip().upper()
    task_key = _ck_task(task_id)

    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_TASK_RESULT_SEC)

    try:
        final_state = await app_graph.ainvoke({"symbol": symbol, "iterations": 0, "task_id": task_id})
        total_dt = time.perf_counter() - t0_total

        report_obj = final_state.get("report")

        payload = {
            "status": "complete",
            "data": {
                "report": report_obj,
                "total_seconds": round(total_dt, 3),
            },
            "debug": final_state.get("debug", {}),
        }

        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)

        # Optional symbol-level cache
        try:
            cache_set(_ck_report(symbol), report_obj, ttl_seconds=TTL_ANALYSIS_REPORT_SEC)
        except Exception:
            pass

        logger.info("[%s] %s: total=%.2fs", task_id, symbol, total_dt)

    except Exception as e:
        total_dt = time.perf_counter() - t0_total
        logger.exception("Analysis failed for %s (%s).", symbol, task_id)

        payload = {
            "status": "failed",
            "data": {"error": str(e), "total_seconds": round(total_dt, 3)},
        }
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
