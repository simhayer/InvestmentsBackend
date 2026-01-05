# routers/v2/analyse_symbol_routes.py
import os
import time
import logging
import asyncio
from typing import TypedDict, List, Dict, Any

from fastapi import BackgroundTasks, HTTPException, APIRouter
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tavily import AsyncTavilyClient

# ✅ Change this import to wherever your function lives
from services.fundamentals.finnhub_fundamentals import fetch_fundamentals  # <-- adjust path

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
    # request context
    symbol: str
    task_id: str

    # data sources
    raw_data: str
    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]

    # specialist drafts
    fundamentals: str
    technicals: str
    risks: str

    # final structured report
    report: Dict[str, Any]

    # loop control
    critique: str
    is_valid: bool
    iterations: int


# ----------------------------
# 3) Clients
# ----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_KEY:
    logger.warning("TAVILY_API_KEY is not set. Tavily calls will fail.")
tavily = AsyncTavilyClient(api_key=TAVILY_KEY or "")


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
# 5) Graph Nodes
# ----------------------------
async def research_node(state: AgentState):
    symbol = state["symbol"]

    async def tav():
        with timed("tavily_search", state):
            results = await tavily.search(
                query=f"latest news, catalysts, and bear case for {symbol}",
                topic="finance",
                search_depth="advanced",
            )
        return str(results)

    async def fin():
        with timed("finnhub_fundamentals", state):
            finres = await fetch_fundamentals(symbol, timeout_s=5.0)
        return finres

    raw_str, finres = await asyncio.gather(tav(), fin())
    return {"raw_data": raw_str, "finnhub_data": finres.data, "finnhub_gaps": finres.gaps}


async def fundamentals_node(state: AgentState):
    symbol = state["symbol"]
    with timed("llm_fundamentals", state):
        prompt = f"""
You are analyzing {symbol}.

FINNHUB DATA (numbers you can trust):
{state.get('finnhub_data', {})}

FINNHUB GAPS:
{state.get('finnhub_gaps', [])}

WEB/NEWS CONTEXT:
{state.get('raw_data', '')}

Task:
Write 3-6 Key Insights as short bullets. Each bullet should reference at least
one concrete metric if available (valuation, margins, growth, debt, cash flow).
Return as plain text bullets.
"""
        res = await llm.ainvoke(prompt)
    return {"fundamentals": res.content}


async def technicals_node(state: AgentState):
    symbol = state["symbol"]
    with timed("llm_technicals", state):
        prompt = f"""
Write the Current Performance for {symbol}.

FINNHUB SNAPSHOT:
{state.get('finnhub_data', {})}

WEB/NEWS CONTEXT:
{state.get('raw_data', '')}

Task:
Write a short paragraph on current performance:
- recent trend / sentiment
- earnings reaction (if present)
- volatility / momentum qualitatively
Do NOT invent RSI values if not present in the data.
"""
        res = await llm.ainvoke(prompt)
    return {"technicals": res.content}


async def risks_node(state: AgentState):
    symbol = state["symbol"]
    with timed("llm_risks", state):
        prompt = f"""
Identify Stock Overflow risks and red flags for {symbol}. Be skeptical.

FINNHUB DATA:
{state.get('finnhub_data', {})}

WEB/NEWS CONTEXT:
{state.get('raw_data', '')}

Task:
List 4-8 risks as short bullets. Avoid generic one-word risks.
Prefer specific risks tied to margins, debt, cash flow, guidance, execution, competition, macro.
"""
        res = await llm.ainvoke(prompt)
    return {"risks": res.content}


async def analyst_structured_node(state: AgentState):
    """
    ✅ Produces the final AnalysisReport directly (no separate formatting call).
    If critic suggests fixes, we feed them back in.
    """
    symbol = state["symbol"]

    structured_llm = llm.with_structured_output(AnalysisReport, method="function_calling")

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE / FIXES TO APPLY:\n{critique}\n" if critique else ""

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
- Use FINNHUB numbers as the source of truth for metrics.
- If FINNHUB has gaps, mention uncertainty in wording (but still fill all fields).

SYMBOL: {symbol}

FINNHUB DATA:
{state.get('finnhub_data', {})}

FINNHUB GAPS:
{state.get('finnhub_gaps', [])}

WEB/NEWS CONTEXT:
{state.get('raw_data', '')}

DRAFT INSIGHTS:
{state.get('fundamentals', '')}

DRAFT PERFORMANCE:
{state.get('technicals', '')}

DRAFT RISKS:
{state.get('risks', '')}
{critique_block}
"""
        report = await structured_llm.ainvoke(prompt)

    return {"report": report.model_dump(), "iterations": int(state.get("iterations", 0)) + 1}


async def critic_node(state: AgentState):
    """
    ✅ Cheap-ish guardrail. Max 1 revision for speed.
    Returns:
      - is_valid: bool
      - critique: string (fixes)
    """
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
    is_clear = "CLEAR" in txt.upper()

    # ✅ cap to 1 revise max (fast)
    iters = int(state.get("iterations", 0))
    if iters >= 1:
        return {"is_valid": True, "critique": txt}

    return {"is_valid": bool(is_clear), "critique": txt}


# ----------------------------
# 6) Build Graph
# ----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("research", research_node)
workflow.add_node("fundamentals", fundamentals_node)
workflow.add_node("technicals", technicals_node)
workflow.add_node("risks", risks_node)
workflow.add_node("analyst", analyst_structured_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("research")

# fan out
workflow.add_edge("research", "fundamentals")
workflow.add_edge("research", "technicals")
workflow.add_edge("research", "risks")

# join into analyst
workflow.add_edge("fundamentals", "analyst")
workflow.add_edge("technicals", "analyst")
workflow.add_edge("risks", "analyst")

workflow.add_edge("analyst", "critic")

workflow.add_conditional_edges(
    "critic",
    lambda x: "end" if x.get("is_valid") else "revise",
    {"revise": "analyst", "end": END},
)

app_graph = workflow.compile()


# ----------------------------
# 7) Storage (pilot)
# ----------------------------
results_db: Dict[str, object] = {}  # "processing" OR {"report":..., "total_seconds":...} OR {"error":...}


# ----------------------------
# 8) Background worker
# ----------------------------
async def run_analysis_task(symbol: str, task_id: str):
    t0_total = time.perf_counter()
    try:
        final_state = await app_graph.ainvoke({"symbol": symbol, "iterations": 0, "task_id": task_id})
        total_dt = time.perf_counter() - t0_total

        logger.info("[%s] %s: total=%.2fs", task_id, symbol, total_dt)

        # final_state["report"] is already a dict
        results_db[task_id] = {"report": final_state.get("report"), "total_seconds": round(total_dt, 3)}
    except Exception as e:
        total_dt = time.perf_counter() - t0_total
        logger.exception("Analysis failed for %s (%s).", symbol, task_id)
        results_db[task_id] = {"error": str(e), "total_seconds": round(total_dt, 3)}


# ----------------------------
# 9) Routes
# ----------------------------
@router.post("/analyze/{symbol}")
async def start_analysis(symbol: str, bg: BackgroundTasks):
    task_id = f"task_{symbol}_{os.urandom(4).hex()}"
    results_db[task_id] = "processing"
    bg.add_task(run_analysis_task, symbol, task_id)
    return {"task_id": task_id, "status": "started"}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    result = results_db.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    if result == "processing":
        return {"status": "processing", "data": None}

    if isinstance(result, dict) and "error" in result:
        return {"status": "failed", "data": result}

    return {"status": "complete", "data": result}
