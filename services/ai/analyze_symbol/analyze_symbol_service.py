# services/ai/analyze_symbol/analyze_symbol_service.py
import os
import time
import json
import logging
import asyncio
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from services.openai.client import llm
from services.vector.vector_store_service import VectorStoreService
from services.cache.cache_backend import cache_set
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
    validate_report,
    compute_data_quality,
)

from .types import AgentState, CoreAnalysis
from .facts_pack_service import (
    build_facts_pack,
    build_key_risks,
    build_current_performance,
    build_price_outlook,
    build_watch_list,
)

try:
    from services.tavily.client import search as tavily_search, compact_results as compact_tavily
except Exception:
    tavily_search = None  # type: ignore
    compact_tavily = None  # type: ignore


logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))
TTL_ANALYSIS_REPORT_SEC = int(os.getenv("TTL_ANALYSIS_REPORT_SEC", "1800"))

# Heavy clients ONCE
VS = VectorStoreService()

FINNHUB_SEM = asyncio.Semaphore(int(os.getenv("FINNHUB_CONCURRENCY", "5")))
SEC_SEM = asyncio.Semaphore(int(os.getenv("SEC_CONCURRENCY", "2")))
TAVILY_SEM = asyncio.Semaphore(int(os.getenv("TAVILY_CONCURRENCY", "2")))


async def _sem_run(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro


def _safe_sorted_earnings(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items or [], key=lambda x: x.get("date") or "9999-99-99")


async def _build_technicals(symbol: str) -> str:
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
    # Keep deterministic text builder where it currently lives
    from services.ai.helpers.analyze_symbol_helpers import build_technicals_text_from_pack
    return build_technicals_text_from_pack(symbol, tech_pack)


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
            return await _sem_run(
                SEC_SEM,
                asyncio.to_thread(
                    get_sec_routed_context,
                    symbol=symbol,
                    state=state,
                    task_id=task_id,
                    vs=VS,
                ),
            )

    async def get_tech():
        with timed("technicals_pack", logger, state=state):
            return await _build_technicals(symbol)

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


async def facts_pack_node(state: AgentState):
    facts_pack = build_facts_pack(
        market_snapshot=state.get("market_snapshot") or {},
        quote=(state.get("finnhub_data", {}).get("quote") or {}),
        peer_ready=(state.get("peer_comparison_ready") or {}),
        earnings_small=_safe_sorted_earnings(state.get("earnings_calendar", []))[:1],
        news_items=state.get("news_items", []),
        sec_risks=state.get("sec_risks", []),
        data_quality_notes=compute_data_quality(
            finnhub_data=state.get("finnhub_data", {}),
            finnhub_gaps=state.get("finnhub_gaps", []),
            sec_context=state.get("sec_context", ""),
            news_count=len(state.get("news_items", [])),
        ),
    )
    return {"facts_pack": facts_pack.model_dump()}

async def analyst_core_node(state: AgentState):
    llm_core = llm.with_structured_output(CoreAnalysis).bind(temperature=0.2)

    facts = state["facts_pack"]

    prompt = f"""
    You are given a FACTS PACK. Treat it as correct.

    Your job:
    - Decide what matters (no filler)
    - Decide what is priced in
    - Explain one place the market may be wrong

    OUTPUT RULES (STRICT)
    - key_insights: EXACTLY 3
    - thesis_points: EXACTLY 3
    - upcoming_catalysts: EXACTLY 3 and distinct
    - Only ONE catalyst may be earnings-related (use facts.events.next_earnings as the window)
    - scenarios: EXACTLY 3 with names: Base, Bull, Bear

    QUALITY RULES
    - Every key_insight must include a real metric or a concrete label from the pack (price levels, YoY %, margin %, leverage flag, trend label, etc.)
    - Evidence MUST be a short plain-English sentence.

    FACTS PACK:
    {json_dumps(facts)}
    """.strip()

    core = await llm_core.ainvoke(prompt)
    return {"core_analysis": core.model_dump()}


def assemble_report_node(state: AgentState):
    core = state["core_analysis"]
    facts = state["facts_pack"]

    peer_ready = state.get("peer_comparison_ready") or {}

    report = {
        **core,
        "symbol": state["symbol"],

        # Python-owned fields
        "current_performance": build_current_performance(facts),
        "key_risks": build_key_risks(facts),
        "price_outlook": build_price_outlook(facts, core),
        "what_to_watch_next": build_watch_list(facts, core),

        # Peer: include raw + computed summary (optional)
        "peer_comparison": peer_ready if peer_ready.get("peers_used") else None,
        "peer_comparison_summary": build_peer_summary(peer_ready) if peer_ready.get("peers_used") else [],

        "data_quality_notes": facts.get("data_quality_notes", []),
    }

    return {"report": report}


# ----------------------------
# Build Graph
# ----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("facts_pack", facts_pack_node)
workflow.add_node("analyst_core", analyst_core_node)
workflow.add_node("assemble", assemble_report_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "facts_pack")
workflow.add_edge("facts_pack", "analyst_core")
workflow.add_edge("analyst_core", "assemble")
workflow.add_edge("assemble", END)

app_graph = workflow.compile()


# ----------------------------
# Task runner (Redis-backed)
# ----------------------------
async def run_analysis_task(symbol: str, task_id: str):
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
