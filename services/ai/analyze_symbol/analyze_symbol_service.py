# services/ai/analyze_symbol_service.py
import os
import time
import logging
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from database import get_db
from services.vector.vector_store_service import VectorStoreService
from utils.common_helpers import timed
from services.tavily.client import search as tavily_search, compact_results as compact_tavily
from services.fundamentals.finnhub_fundamentals import fetch_fundamentals_cached
from services.cache.cache_backend import cache_get, cache_set
from .types import AnalysisReport, AgentState

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

def _preview(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")

TTL_FUNDAMENTALS_SEC = int(os.getenv("TTL_FUNDAMENTALS_SEC", "600"))  # 10m
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))              # 15m
TTL_TASK_RESULT_SEC = int(os.getenv("TTL_TASK_RESULT_SEC", "3600"))   # 1h

def _ck_fund(symbol: str) -> str:
    return f"ANALYZE:FUND:{(symbol or '').strip().upper()}"

def _ck_tav(symbol: str) -> str:
    return f"ANALYZE:TAV:{(symbol or '').strip().upper()}"

def _ck_task(task_id: str) -> str:
    return f"ANALYZE:TASK:{(task_id or '').strip()}"

# ----------------------------
# Graph Nodes
# ----------------------------
async def research_node(state: AgentState):
    symbol = state.get("symbol") or ""

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

    raw_str = ""
    tav_key = _ck_tav(symbol)
    cached_t = cache_get(tav_key)
    if isinstance(cached_t, str):
        raw_str = cached_t
    elif isinstance(cached_t, dict) and "raw" in cached_t and isinstance(cached_t["raw"], str):
        raw_str = cached_t["raw"]
    else:
        if tavily_search is None:
            raw_str = ""
        else:
            try:
                with timed("tavily_search", state, logger):
                    results = await tavily_search(
                        query=f"latest news, catalysts, and bear case for {symbol}",
                        max_results=8,
                        include_answer=False,
                        include_raw_content=False,
                        search_depth="advanced",
                    )
                raw_str = compact_tavily(results)
                cache_set(tav_key, {"raw": raw_str}, ttl_seconds=TTL_TAVILY_SEC)
            except Exception:
                # do not fail the whole analysis if tavily fails
                raw_str = ""

    sec_context = ""
    sec_debug: Dict[str, Any] = {"count": 0, "chunks": []}
    try:
        db_gen = get_db()
        db = next(db_gen)
        vector_service = VectorStoreService()

        sec_chunks = vector_service.get_context_for_analysis(
            db=db,
            symbol=symbol,
            query="business model, key risks, liquidity, margins, guidance",
            limit=8,
        )

        logger.info("[%s] %s: sec_ctx=%d", state.get("task_id", "no_task"), symbol, len(sec_chunks))

        sec_debug["count"] = len(sec_chunks)
        sec_debug["chunks"] = [
            {
                "score": c.get("score"),
                "metadata": c.get("metadata"),
                "content_preview": (c.get("content") or "")[:240],
            }
            for c in sec_chunks
        ]

        if sec_chunks:
            sec_context = "\n".join(
                f"- ({c.get('metadata', {}).get('form_type','?')} {c.get('metadata', {}).get('filed_date','?')} | {c.get('metadata', {}).get('filing_id','?')}) {c['content']}"
                for c in sec_chunks
            )[:8000]
        else:
            logger.info("[%s] %s: no sec context found", state.get("task_id", "no_task"), symbol)

    except Exception as e:
        logger.warning("SEC vector lookup failed for %s: %s", symbol, e)
    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass

    debug = {
        "symbol": symbol,
        "finnhub": {
            "gaps": finnhub_gaps,
            "top_keys": sorted(list(finnhub_data.keys()))[:40],
        },
        "tavily": {
            "chars": len(raw_str or ""),
            "preview": _preview(raw_str, 280),
            "used": bool(raw_str),
        },
        "sec": {
            "chars": len(sec_context or ""),
            **sec_debug,
        },
    }
    return {"raw_data": raw_str, "finnhub_data": finnhub_data, "finnhub_gaps": finnhub_gaps, "sec_context": sec_context, "debug": debug}


async def drafts_node(state: AgentState):
    """
    Runs fundamentals/technicals/risks drafts in parallel.
    This avoids LangGraph join/fanout edge weirdness and guarantees analyst runs once.
    """
    symbol = state.get("symbol") or ""

    finnhub_data = state.get("finnhub_data", {})
    finnhub_gaps = state.get("finnhub_gaps", [])
    raw_data = state.get("raw_data", "")
    sec_context = state.get("sec_context", "")

    async def do_fundamentals():
        with timed("llm_fundamentals", state, logger):
            prompt = f"""
                You are analyzing {symbol}.

                SEC FILING CONTEXT (authoritative, may be empty):
                {sec_context}

                FINNHUB DATA (numbers you can trust):
                {finnhub_data}

                FINNHUB GAPS:
                {finnhub_gaps}

                WEB/NEWS CONTEXT (may be empty):
                {raw_data}

                Task:
                Write 3-6 Key Insights as short bullets.
                Each bullet should reference at least one concrete metric if available.
                If a metric is missing, say "Not available" instead of guessing.
                Return as plain text bullets.
                """
            res = await llm.ainvoke(prompt)
        return res.content or ""

    async def do_technicals():
        with timed("llm_technicals", state, logger):
            prompt = f"""
                Write the Current Performance for {symbol}.

                SEC FILING CONTEXT (authoritative, may be empty):
                {sec_context}

                FINNHUB SNAPSHOT:
                {finnhub_data}

                WEB/NEWS CONTEXT (may be empty):
                {raw_data}

                Task:
                Write a short paragraph on current performance:
                - recent trend / sentiment
                - earnings reaction (if present)
                - volatility / momentum qualitatively
                Do NOT invent RSI or indicator values if not present in the data.
                """
            res = await llm.ainvoke(prompt)
        return res.content or ""

    async def do_risks():
        with timed("llm_risks", state, logger):
            prompt = f"""
                Identify Stock Overflow risks and red flags for {symbol}. Be skeptical.

                SEC FILING CONTEXT (authoritative, may be empty):
                {sec_context}

                FINNHUB DATA:
                {finnhub_data}

                WEB/NEWS CONTEXT (may be empty):
                {raw_data}

                Task:
                List 4-8 risks as short bullets. Avoid generic one-word risks.
                Prefer specific risks tied to margins, debt, cash flow, guidance, execution, competition, macro.
                If data is missing, describe the uncertainty clearly.
                """
            res = await llm.ainvoke(prompt)
        return res.content or ""

    fundamentals, technicals, risks = await asyncio.gather(
        do_fundamentals(), do_technicals(), do_risks()
    )

    return {"fundamentals": fundamentals, "technicals": technicals, "risks": risks}

async def analyst_structured_node(state: AgentState):
    """
    Produces final AnalysisReport (structured output).
    If critic suggests fixes, we feed them back in.
    """
    symbol = state.get("symbol") or ""
    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling")

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE / FIXES TO APPLY:\n{critique}\n" if critique and critique.upper() != "CLEAR" else ""

    finnhub_data = state.get("finnhub_data", {})
    finnhub_gaps = state.get("finnhub_gaps", [])
    raw_data = state.get("raw_data", "")
    sec_context = state.get("sec_context", "")

    with timed("llm_analyst_structured", state, logger):
        prompt = f"""
            Return a JSON object that matches the AnalysisReport schema EXACTLY.

            Rules:
            - key_insights: list of short bullets (3-6)
            - stock_overflow_risks: list of short bullets (4-8)
            - current_performance: short paragraph
            - price_outlook: must include base/bull/bear logic in text
            - recommendation: Buy / Hold / Sell (capitalize first letter)
            - confidence: 0.0 to 1.0
            - is_priced_in: true/false

            Source rules (STRICT):
            - FINNHUB is the only source for precise financial metrics and valuation ratios.
            - SEC filings are authoritative for risks, business model, dependencies, margin variability, legal/regulatory, liquidity narrative.
            - WEB/NEWS is contextual only (sentiment/catalysts) and MUST be treated as non-authoritative.

            No-target rule (STRICT):
            - DO NOT output exact "fair value" / price target numbers unless they appear in FINNHUB DATA.
            - If WEB mentions targets/fair value, only say "external commentary suggests valuation concerns" without numbers.

            Grounding requirements:
            - key_insights must include >=1 insight grounded in SEC filings.
            - stock_overflow_risks must include >=2 risks grounded in SEC filings.
            - If something is missing, say "Not available" rather than guessing.

            Confidence guidance:
            - If FINNHUB has gaps -> cap confidence at 0.55
            - Else if BOTH SEC context and WEB context are empty -> cap confidence at 0.55
            - Else allow up to 0.75, but keep it modest if conclusions rely mostly on WEB context.

            Recommendation must match the outlook and risks.

            SYMBOL: {symbol}

            SEC FILING CONTEXT (authoritative, may be empty):
            {sec_context}

            FINNHUB DATA:
            {finnhub_data}

            FINNHUB GAPS:
            {finnhub_gaps}

            WEB/NEWS CONTEXT:
            {raw_data}

            DRAFT INSIGHTS:
            {state.get('fundamentals', '')}

            DRAFT PERFORMANCE:
            {state.get('technicals', '')}

            DRAFT RISKS:
            {state.get('risks', '')}
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

    with timed("llm_critic", state, logger):
        prompt = f"""
            You are a strict reviewer. Check this report for:
            - missing key risks
            - generic/fluffy claims
            - contradictions between metrics and conclusion
            - recommendation not matching reasoning
            - confidence too high/low given uncertainty

            If it is good enough to ship, respond exactly: CLEAR
            Otherwise respond with 3-7 specific fixes.

            REPORT JSON:
            {state.get('report', {})}
            """
        res = await llm.ainvoke(prompt)

    txt = (res.content or "").strip()
    is_clear = txt.upper() == "CLEAR"

    # If already revised once, stop revising (ship whatever we have)
    if iters >= 1:
        return {"is_valid": True, "critique": "" if is_clear else txt}

    return {"is_valid": bool(is_clear), "critique": "" if is_clear else txt}


# ----------------------------
# 8) Build Graph
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
# 9) Task runner (Redis-backed)
# ----------------------------
async def run_analysis_task(symbol: str, task_id: str):
    t0_total = time.perf_counter()
    task_key = _ck_task(task_id)

    # mark processing
    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_TASK_RESULT_SEC)

    try:
        final_state = await app_graph.ainvoke({"symbol": symbol, "iterations": 0, "task_id": task_id})
        total_dt = time.perf_counter() - t0_total

        payload = {
            "status": "complete",
            "data": {
                "report": final_state.get("report"),
                "total_seconds": round(total_dt, 3),
            },
            "debug": final_state.get("debug", {}),
        }
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
        logger.info("[%s] %s: total=%.2fs", task_id, symbol, total_dt)

    except Exception as e:
        total_dt = time.perf_counter() - t0_total
        logger.exception("Analysis failed for %s (%s).", symbol, task_id)

        payload = {
            "status": "failed",
            "data": {"error": str(e), "total_seconds": round(total_dt, 3)},
        }
        cache_set(task_key, payload, ttl_seconds=TTL_TASK_RESULT_SEC)
