# routers/v2/analyse_symbol_routes.py
import os
import time
import logging
import asyncio
from typing import TypedDict, List, Dict, Any, Optional

from fastapi import BackgroundTasks, HTTPException, APIRouter
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

try:
    from tavily import AsyncTavilyClient
except Exception:
    AsyncTavilyClient = None  # type: ignore

from services.fundamentals.finnhub_fundamentals import fetch_fundamentals_cached
from services.cache.cache_backend import cache_get, cache_set

router = APIRouter()

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ----------------------------
# 1) Output schema
# ----------------------------
class AnalysisReport(BaseModel):
    symbol: str
    key_insights: List[str] = Field(description="Critical fundamental highlights")
    current_performance: str = Field(description="Technical and price action analysis")
    stock_overflow_risks: List[str] = Field(description="Red flags and assessment of risks")
    price_outlook: str = Field(description="Deeply reasoned AI outlook balancing bull/bear cases")
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_priced_in: bool = False


# ----------------------------
# 2) Graph state
# ----------------------------
class AgentState(TypedDict, total=False):
    symbol: str
    task_id: str

    raw_data: str
    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]

    fundamentals: str
    technicals: str
    risks: str

    report: Dict[str, Any]

    critique: str
    is_valid: bool
    iterations: int


# ----------------------------
# 3) Clients
# ----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
tavily: Optional[Any] = None
if AsyncTavilyClient and TAVILY_KEY:
    tavily = AsyncTavilyClient(api_key=TAVILY_KEY)
elif not TAVILY_KEY:
    logger.warning("TAVILY_API_KEY is not set. Tavily will be skipped.")


# ----------------------------
# 4) Timing helper
# ----------------------------
def timed(name: str, state: AgentState):
    symbol = state.get("symbol", "NA")
    task_id = state.get("task_id", "no_task")

    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            logger.info("[%s] %s: %s=%.2fs", task_id, symbol, name, dt)
            return False

    return _T()


# ----------------------------
# 5) Cache keys / TTLs
# ----------------------------
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
# 6) Tavily compaction (smaller context for LLM)
# ----------------------------
def compact_tavily(results: Any, limit: int = 6) -> str:
    """
    Tavily returns a dict-like structure. We keep only the top items (title/url/snippet).
    If it's already a string, return as-is (but truncated).
    """
    if results is None:
        return ""

    if isinstance(results, str):
        return results[:8000]

    if isinstance(results, dict):
        items = results.get("results") or results.get("data") or []
        if not isinstance(items, list):
            return str(results)[:8000]

        out_lines: List[str] = []
        for r in items[: max(1, limit)]:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or r.get("snippet") or "").strip()
            if content:
                content = content.replace("\n", " ").strip()
            line = f"- {title}\n  {url}\n  {content}"
            out_lines.append(line)

        return "\n".join(out_lines)[:8000]

    return str(results)[:8000]


# ----------------------------
# 7) Graph Nodes
# ----------------------------
async def research_node(state: AgentState):
    symbol = state["symbol"]

    # ---- fundamentals cached
    fin_key = _ck_fund(symbol)
    cached_f = cache_get(fin_key)
    if isinstance(cached_f, dict) and "data" in cached_f and "gaps" in cached_f:
        finnhub_data = cached_f.get("data") or {}
        finnhub_gaps = cached_f.get("gaps") or []
    else:
        with timed("finnhub_fundamentals", state):
            finres = await fetch_fundamentals_cached(symbol, timeout_s=5.0)
        finnhub_data = finres.data
        finnhub_gaps = finres.gaps
        cache_set(fin_key, {"data": finnhub_data, "gaps": finnhub_gaps}, ttl_seconds=TTL_FUNDAMENTALS_SEC)

    # ---- tavily cached (optional)
    raw_str = ""
    tav_key = _ck_tav(symbol)
    cached_t = cache_get(tav_key)
    if isinstance(cached_t, str):
        raw_str = cached_t
    elif isinstance(cached_t, dict) and "raw" in cached_t and isinstance(cached_t["raw"], str):
        raw_str = cached_t["raw"]
    else:
        if tavily is None:
            raw_str = ""
        else:
            try:
                with timed("tavily_search", state):
                    results = await tavily.search(
                        query=f"latest news, catalysts, and bear case for {symbol}",
                        topic="finance",
                        search_depth="advanced",
                    )
                raw_str = compact_tavily(results)
                cache_set(tav_key, {"raw": raw_str}, ttl_seconds=TTL_TAVILY_SEC)
            except Exception:
                # do not fail the whole analysis if tavily fails
                raw_str = ""

    return {"raw_data": raw_str, "finnhub_data": finnhub_data, "finnhub_gaps": finnhub_gaps}


async def drafts_node(state: AgentState):
    """
    Runs fundamentals/technicals/risks drafts in parallel.
    This avoids LangGraph join/fanout edge weirdness and guarantees analyst runs once.
    """
    symbol = state["symbol"]

    finnhub_data = state.get("finnhub_data", {})
    finnhub_gaps = state.get("finnhub_gaps", [])
    raw_data = state.get("raw_data", "")

    async def do_fundamentals():
        with timed("llm_fundamentals", state):
            prompt = f"""
You are analyzing {symbol}.

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
        with timed("llm_technicals", state):
            prompt = f"""
Write the Current Performance for {symbol}.

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
        with timed("llm_risks", state):
            prompt = f"""
Identify Stock Overflow risks and red flags for {symbol}. Be skeptical.

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
    symbol = state["symbol"]
    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling")

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE / FIXES TO APPLY:\n{critique}\n" if critique and critique.upper() != "CLEAR" else ""

    finnhub_data = state.get("finnhub_data", {})
    finnhub_gaps = state.get("finnhub_gaps", [])
    raw_data = state.get("raw_data", "")

    with timed("llm_analyst_structured", state):
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
- Use FINNHUB numbers as source of truth.
- If FINNHUB has gaps OR web context is empty, keep confidence modest (<= 0.55).
- If something is missing, say "Not available" rather than guessing.
- Recommendation must match the outlook and risks.

SYMBOL: {symbol}

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

    with timed("llm_critic", state):
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


# ----------------------------
# 10) Routes
# ----------------------------
@router.post("/analyze/{symbol}")
async def start_analysis(symbol: str, bg: BackgroundTasks):
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise HTTPException(status_code=400, detail="Missing symbol")

    task_id = f"task_{clean_symbol}_{os.urandom(4).hex()}"
    task_key = _ck_task(task_id)

    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_TASK_RESULT_SEC)

    # FastAPI/Starlette can run async background tasks; this is OK.
    bg.add_task(run_analysis_task, clean_symbol, task_id)

    return {"task_id": task_id, "status": "started"}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task_id")

    task_key = _ck_task(task_id)
    result = cache_get(task_key)

    if not result:
        raise HTTPException(status_code=404, detail="Task not found (expired or invalid)")

    if isinstance(result, dict):
        # expected shape: {"status": "...", "data": ...}
        return result

    # fallback if something weird got stored
    return {"status": "failed", "data": {"error": "Invalid task payload"}}
