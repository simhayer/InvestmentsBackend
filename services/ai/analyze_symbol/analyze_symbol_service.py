# services/ai/analyze_symbol_service.py
import os
import time
import json
import logging
import asyncio
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from services.openai.client import llm, llm_mini  # llm = gpt-4o, llm_mini = gpt-4o-mini
from services.vector.vector_store_service import VectorStoreService
from utils.common_helpers import timed

try:
    from services.tavily.client import search as tavily_search, compact_results as compact_tavily
except Exception:
    tavily_search = None  # type: ignore
    compact_tavily = None  # type: ignore

from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.finnhub.peer_benchmark_service import fetch_peer_benchmark_cached, build_peer_summary, build_peer_positioning, build_peer_comparison_ready
from services.cache.cache_backend import cache_set
from services.ai.technicals_pack import build_technical_pack, compact_tech_pack
from services.finnhub.finnhub_news_service import get_company_news_cached, shrink_news_items
from services.finnhub.finnhub_calender_service import get_earnings_calendar_compact_cached

from services.ai.helpers.analyze_symbol_helpers import (
    compute_market_snapshot,
    fetch_history_points,
    normalize_symbol,
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
    validate_report,
    compute_data_quality,
)

from .types import AnalysisReport, AgentState

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))  # 10m
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))  # 15m
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))  # 1h
TTL_ANALYSIS_REPORT_SEC = int(os.getenv("TTL_ANALYSIS_REPORT_SEC", "1800"))  # 30m per symbol

# Heavy clients ONCE
VS = VectorStoreService()

# Optional: limit fan-out so you don’t get throttled under load
FINNHUB_SEM = asyncio.Semaphore(int(os.getenv("FINNHUB_CONCURRENCY", "5")))
SEC_SEM = asyncio.Semaphore(int(os.getenv("SEC_CONCURRENCY", "2")))
TAVILY_SEM = asyncio.Semaphore(int(os.getenv("TAVILY_CONCURRENCY", "2")))


async def _sem_run(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro


def _safe_sorted_earnings(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items or [], key=lambda x: x.get("date") or "9999-99-99")

async def _build_technicals(symbol: str) -> str:
    # deterministic technicals text (still needs history IO)
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


def _trim_inputs_for_structured(state: AgentState) -> Dict[str, Any]:
    finnhub_data = state.get("finnhub_data", {}) or {}
    normalized = finnhub_data.get("normalized") or {}

    profile_small = trim_profile(finnhub_data.get("profile") or {})
    quote_small = trim_quote(finnhub_data.get("quote") or {})
    market_snapshot_small = trim_market_snapshot(state.get("market_snapshot", {}) or {})

    earnings_calendar = _safe_sorted_earnings(state.get("earnings_calendar", []) or [])
    earnings_small = trim_earnings_calendar(earnings_calendar, max_items=2)
    next_date = earnings_small[0].get("date") if earnings_small else "Unknown"

    # Reduce SEC further for speed (you can tune this)
    sec_bus_small = trim_sec_chunks(state.get("sec_business", []) or [], max_chunks=1, max_chars_each=650)
    sec_risks_small = trim_sec_chunks(state.get("sec_risks", []) or [], max_chunks=1, max_chars_each=650)
    sec_mda_small = trim_sec_chunks(state.get("sec_mda", []) or [], max_chunks=1, max_chars_each=650)

    news_items = state.get("news_items", []) or []
    news_small = trim_news_items(news_items, max_items=4, max_chars_each=200)

    finnhub_gaps = state.get("finnhub_gaps", []) or []
    data_quality_notes = compute_data_quality(
        finnhub_data=finnhub_data,
        finnhub_gaps=finnhub_gaps,
        sec_context=state.get("sec_context", "") or "",
        news_count=len(news_items),
    )

    peer_ready = state.get("peer_comparison_ready") or {}
    if not peer_ready:
        peer_ready = build_peer_comparison_ready(state.get("peer_benchmark", {}) or {})

    peer_positioning = build_peer_positioning(peer_ready)

    technicals_text = shorten_text(state.get("technicals", "") or "", 650)

    critique = (state.get("critique") or "").strip()
    critique_block = (
        f"\nCRITIQUE FIXES:\n{shorten_text(critique, 900)}\n"
        if critique and critique.upper() != "CLEAR"
        else ""
    )

    return {
        "symbol": (state.get("symbol") or "").strip().upper(),
        "normalized": normalized,
        "profile_small": profile_small,
        "quote_small": quote_small,
        "market_snapshot_small": market_snapshot_small,
        "earnings_small": earnings_small,
        "next_date": str(next_date) if next_date else "Unknown",
        "sec_bus_small": sec_bus_small,
        "sec_risks_small": sec_risks_small,
        "sec_mda_small": sec_mda_small,
        "news_small": news_small,
        "technicals_text": technicals_text,
        "peer_ready": peer_ready,
        "data_quality_notes": data_quality_notes,
        "critique_block": critique_block,
        "peer_positioning": peer_positioning,
    }


def _build_structured_prompt(trimmed: Dict[str, Any]) -> str:
    symbol = trimmed["symbol"]
    next_date = trimmed["next_date"]
    critique_block = trimmed["critique_block"]

    return f"""
Return a JSON object that matches the AnalysisReport schema EXACTLY.
Do not include any fields not defined in the schema.

Definitive truth: next earnings date is {next_date}.

====================
CORE RULES
====================

STYLE
- Write like a buy-side analyst.
- Be specific. Avoid filler and generic statements.

STRUCTURE (STRICT)
- key_insights: EXACTLY 3 items.
- key_risks: EXACTLY 5 items.
- thesis_points: EXACTLY 3 items.
- upcoming_catalysts: EXACTLY 3 items.
- scenarios: EXACTLY 3 (Base, Bull, Bear).

CONTENT RULES
- Each key_insight MUST include: (1) a metric, (2) evidence, (3) implication.
- unified_thesis MUST be concise and coherent.
- is_priced_in must be boolean.
- pricing_assessment MUST be present and non-empty.
- If confidence ≥ 0.6, market_edge MUST be present.

PEERS
- You may reference peer positioning qualitatively (cheap/expensive, strong/weak).
- DO NOT restate or reproduce numerical peer metrics.

CATALYSTS
- If earnings-related, window MUST be "{next_date}".

CONFIDENCE
- Must be defensible.
- If data_quality_notes are non-empty, confidence should generally be < 0.8.

{critique_block}

====================
INPUTS
====================

SYMBOL:
{symbol}

FINNHUB NORMALIZED:
{json_dumps(trimmed["normalized"])}

PROFILE:
{json_dumps(trimmed["profile_small"])}

QUOTE:
{json_dumps(trimmed["quote_small"])}

MARKET SNAPSHOT:
{json_dumps(trimmed["market_snapshot_small"])}

EARNINGS:
{json_dumps(trimmed["earnings_small"])}

SEC BUSINESS:
{json_dumps(trimmed["sec_bus_small"])}

SEC RISKS:
{json_dumps(trimmed["sec_risks_small"])}

SEC MD&A:
{json_dumps(trimmed["sec_mda_small"])}

NEWS:
{json_dumps(trimmed["news_small"])}

TECHNICALS (deterministic):
{trimmed["technicals_text"]}

PEER POSITIONING (qualitative only):
{json_dumps(trimmed["peer_positioning"])}

DATA QUALITY NOTES (copy verbatim):
{json_dumps(trimmed["data_quality_notes"])}

IMPORTANT:
If any rule above is violated, rewrite the JSON before finalizing.
""".strip()


def _build_critic_prompt(issues: List[str], report_obj: Dict[str, Any]) -> str:
    small_payload = {
        "issues": issues,
        "recommendation": report_obj.get("recommendation"),
        "confidence": report_obj.get("confidence"),
        "is_priced_in": report_obj.get("is_priced_in"),
        "pricing_assessment": report_obj.get("pricing_assessment"),
        "upcoming_catalysts": report_obj.get("upcoming_catalysts"),
        "scenarios": report_obj.get("scenarios"),
        "market_edge": report_obj.get("market_edge"),
        "thesis_points": report_obj.get("thesis_points"),
        "key_insights": report_obj.get("key_insights"),
    }

    return f"""
You are a strict reviewer. Fix the report by addressing ONLY the listed issues.
Return 3–7 specific fix instructions. Do not add new sections.

ISSUES:
{json.dumps(issues, ensure_ascii=False)}

PAYLOAD:
{json.dumps(small_payload, ensure_ascii=False)}

If everything is fixed already, respond exactly: CLEAR
""".strip()


# ----------------------------
# Graph Nodes
# ----------------------------

async def research_node(state: AgentState):
    symbol = normalize_symbol(state.get("symbol") or "")
    task_id = state.get("task_id", "no_task")

    async def get_fund():
        return await get_fundamentals_with_cache(
            symbol=symbol,
            state=state,
            ttl_seconds=TTL_FUNDAMENTALS_SEC,
            fetch_fundamentals_cached=fetch_fundamentals_cached,
        )

    async def get_earnings():
        with timed("finnhub_earnings_calendar", logger, state=state):
            items = await _sem_run(
                FINNHUB_SEM,
                get_earnings_calendar_compact_cached(
                    symbol=symbol,
                    window_days=120,
                    limit=6,
                    international=False,
                ),
            )
        return _safe_sorted_earnings(items)

    async def get_peers():
        with timed("finnhub_peer_benchmark", logger, state=state):
            res = await _sem_run(FINNHUB_SEM, fetch_peer_benchmark_cached(symbol, timeout_s=5.0))
        return (res.data or {}), (res.gaps or [])

    async def get_news():
        # Tavily fallback is inside helper; helper should be caching tavily results already
        with timed("finnhub_news", logger, state=state):
            return await get_news_with_optional_tavily_fallback(
                symbol=symbol,
                state=state,
                ttl_tavily_seconds=TTL_TAVILY_SEC,
                get_company_news_cached=get_company_news_cached,
                tavily_search=tavily_search,
                compact_tavily=compact_tavily,
            )

    async def get_sec():
        with timed("sec_vector_routing", logger, state=state):
            return await _sem_run(SEC_SEM, asyncio.to_thread(
                get_sec_routed_context,
                symbol=symbol,
                state=state,
                task_id=task_id,
                vs=VS,
            ))

    async def get_tech():
        with timed("technicals_pack", logger, state=state):
            return await _build_technicals(symbol)

    # Fan-out research + technicals (these are all IO-bound)
    (finnhub_data, finnhub_gaps), earnings_calendar, (peer_benchmark, peer_gaps), (news_items, raw_str, used_tavily), (sec_context, sec_business, sec_risks, sec_mda, sec_debug), technicals_text = await asyncio.gather(
        get_fund(),
        get_earnings(),
        get_peers(),
        get_news(),
        get_sec(),
        get_tech(),
    )

    market_snapshot = compute_market_snapshot(finnhub_data)
    peer_ready = build_peer_comparison_ready(peer_benchmark)

    debug = {
        "symbol": symbol,
        "news": {"count": len(news_items), "used_tavily": used_tavily},
        "sec": {"total_chunks": sec_debug.get("count", 0), "routed": True},
        "earnings_calendar": {"preview": earnings_calendar[:2] if earnings_calendar else []},
        "peers": {"count": len(peer_ready.get("peers_used") or []), "gaps": peer_gaps},
    }

    return {
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
    }


async def analyst_structured_node(state: AgentState):
    """
    Produces AnalysisReport (structured).
    No drafts node: this is now the main thinking step.
    """
    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling").bind(temperature=0.2)

    trimmed = _trim_inputs_for_structured(state)
    prompt = _build_structured_prompt(trimmed)

    with timed(
        "llm_analyst_structured",
        logger,
        state=state,
        tags={"prompt_kb": round(len(prompt) / 1024, 1)},
    ):
        report = await structured_llm.ainvoke(prompt)

    report_obj = report.model_dump()

    # Force canonical peer_comparison copy if peers exist
    peer_ready = trimmed.get("peer_ready") or {}

    if peer_ready.get("peers_used"):
        report_obj["peer_comparison"] = peer_ready
        report_obj["peer_comparison_summary"] = build_peer_summary(peer_ready)

    iters = int(state.get("iterations", 0)) + 1
    return {"report": report_obj, "iterations": iters}


async def critic_node(state: AgentState):
    report_obj = state.get("report", {}) or {}

    earnings_calendar = _safe_sorted_earnings(state.get("earnings_calendar", []) or [])
    raw_next = earnings_calendar[0].get("date") if earnings_calendar else None
    next_date = str(raw_next) if raw_next else "Unknown"

    finnhub_gaps = state.get("finnhub_gaps", []) or []
    issues = validate_report(report_obj, next_earnings_date=next_date, finnhub_gaps=finnhub_gaps)

    if not issues:
        return {"is_valid": True, "critique": "CLEAR"}

    logger.info("Report validation found %d issues for %s", len(issues), state.get("symbol", ""))

    prompt = _build_critic_prompt(issues, report_obj)
    with timed("llm_critic_mini", logger, state=state, tags={"prompt_kb": round(len(prompt) / 1024, 1)}):
        res = await llm_mini.ainvoke(prompt)

    txt = (res.content or "").strip()
    return {"is_valid": txt.upper() == "CLEAR", "critique": "" if txt.upper() == "CLEAR" else txt}


# ----------------------------
# Build Graph (drafts removed)
# ----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analyst", analyst_structured_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "analyst")
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
