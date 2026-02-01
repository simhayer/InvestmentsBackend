from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from models.user_onboarding_profile import UserOnboardingProfile
from services.ai.chat_agent.tools import ToolContext

from .memory_cache import (
    CHAT_ENTITIES_MAX,
    CHAT_MAX_HISTORY,
    append_chat_history,
    load_memory_snapshot,
    save_recent_entities,
    save_thread_summary,
    update_thread_summary,
)
from .state_models import (
    BudgetConfig,
    DataRequirementsPlan,
    GraphState,
    GraphStateDict,
    IntentType,
    RouterOutput,
    MemorySnapshot,
    RequestContext,
    SynthesisOutput,
    ToolBudget,
    ToolError,
    ToolResult,
)
from .tool_executor import ToolCallSpec, ToolExecutor

logger = logging.getLogger("decisioning_graph")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PLANNER_PROMPT_PATH = PROMPT_DIR / "planner_v1.txt"
SYNTHESIS_PROMPT_PATH = PROMPT_DIR / "synthesis_v1.txt"
ROUTER_PROMPT_PATH = PROMPT_DIR / "router_v1.txt"
SYNTHESIS_STREAM_PROMPT_PATH = PROMPT_DIR / "synthesis_stream_v1.txt"
NO_TOOLS_SYNTHESIS_PROMPT_PATH = PROMPT_DIR / "no_tools_synthesis_v1.txt"

INGEST_MODEL = os.getenv("OPENAI_INGEST_MODEL", os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini"))
SMALLTALK_MODEL = os.getenv("OPENAI_SMALLTALK_MODEL", os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini"))
PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
SYNTHESIS_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini"))

DEFAULT_TOOL_TIMEOUT_S = float(os.getenv("CHAT_TOOL_TIMEOUT_SEC", "6"))
DEFAULT_GLOBAL_TIMEOUT_S = float(os.getenv("CHAT_TOOL_GLOBAL_TIMEOUT_SEC", "12"))

DATA_TYPE_TO_TOOL = {
    "portfolio_context": "get_portfolio_context",
    "fundamentals": "get_fundamentals",
    "news": "get_news",
    "sec_snippets": "get_sec_snippets",
    "web_search": "get_web_search",
}

SEC_SECTIONS = {"risk", "mda", "business", "general"}
FUNDAMENTAL_TERMS = (
    "revenue",
    "eps",
    "earnings",
    "net income",
    "profit",
    "margin",
    "cash flow",
    "guidance",
    "ebit",
    "ebitda",
    "free cash flow",
    "balance sheet",
    "income statement",
)

def _sanitize_web_query(q: Optional[str]) -> Optional[str]:
    if not q:
        return None
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < 3:
        return None
    return q[:160] 

def _load_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _log_json(payload: Dict[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, separators=(",", ":")))
    except Exception:
        logger.info(str(payload))


def _log_step(event: str, node: str, state: Dict[str, Any]) -> None:
    _log_json(
        {
            "event": event,
            "node": node,
            "trace_id": state.get("trace_id"),
            "turn_id": state.get("turn_id"),
        }
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _normalize_list(values: List[str]) -> List[str]:
    out = []
    for item in values:
        if not item:
            continue
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _normalize_request_context(ctx: RequestContext) -> RequestContext:
    tickers = _normalize_list([t.upper() for t in (ctx.tickers or []) if isinstance(t, str)])
    sections = _normalize_list(ctx.requested_sections or [])
    return RequestContext(
        intent=ctx.intent,
        tickers=tickers,
        needs_portfolio=ctx.needs_portfolio,
        needs_recency=ctx.needs_recency,
        requested_sections=sections or None,
        output_style=ctx.output_style,
        risk_flags=set(ctx.risk_flags or []),
        timeframe=ctx.timeframe,
        user_constraints=ctx.user_constraints,
    )


def _as_request_context(value: Any) -> RequestContext:
    if isinstance(value, RequestContext):
        return value
    if isinstance(value, dict):
        return RequestContext.model_validate(value)
    return RequestContext()


def _as_memory(value: Any) -> MemorySnapshot:
    if isinstance(value, MemorySnapshot):
        return value
    if isinstance(value, dict):
        return MemorySnapshot.model_validate(value)
    return MemorySnapshot()


def _as_plan(value: Any) -> DataRequirementsPlan:
    if isinstance(value, DataRequirementsPlan):
        return value
    if isinstance(value, dict):
        return DataRequirementsPlan.model_validate(value)
    return DataRequirementsPlan()


def _as_budgets(value: Any) -> BudgetConfig:
    if isinstance(value, BudgetConfig):
        return value
    if isinstance(value, dict):
        return BudgetConfig.model_validate(value)
    return BudgetConfig()


def _as_tool_results(value: Any) -> List[ToolResult]:
    if not value:
        return []
    results: List[ToolResult] = []
    for item in value:
        if isinstance(item, ToolResult):
            results.append(item)
            continue
        if isinstance(item, dict):
            results.append(ToolResult.model_validate(item))
    return results


def _load_user_profile(db: Optional[Session], user_id: Any) -> Dict[str, Any]:
    if db is None or user_id is None:
        return {}
    profile = (
        db.query(UserOnboardingProfile)
        .filter(UserOnboardingProfile.user_id == user_id)
        .first()
    )
    if not profile:
        return {}
    return {
        "time_horizon": profile.time_horizon,
        "primary_goal": profile.primary_goal,
        "risk_level": profile.risk_level,
        "experience_level": profile.experience_level,
        "age_band": profile.age_band,
        "country": profile.country,
        "asset_preferences": profile.asset_preferences,
        "style_preference": profile.style_preference,
        "notification_level": profile.notification_level,
        "notes": profile.notes,
    }


def _allowed_data_types(intent: IntentType) -> List[str]:
    if intent == "portfolio_q":
        return ["portfolio_context", "news", "fundamentals", "sec_snippets"]
    if intent == "single_stock_q":
        return ["fundamentals", "news", "sec_snippets", "web_search"]
    if intent == "news_q":
        return ["news", "fundamentals", "sec_snippets", "web_search"]
    if intent == "sec_q":
        return ["sec_snippets", "fundamentals", "web_search"]
    if intent == "education_q":
        return ["web_search"]
    return []


def _sanitize_data_requirements(plan: DataRequirementsPlan, allowed: List[str]) -> DataRequirementsPlan:
    required = [item for item in plan.required_data if item in allowed]
    optional = [item for item in plan.optional_data if item in allowed and item not in required]
    sec_sections = [section for section in plan.sec_sections if section in SEC_SECTIONS]

    return DataRequirementsPlan(
        required_data=required,
        optional_data=optional,
        sec_sections=sec_sections,
        notes=plan.notes,
        web_search_query=_sanitize_web_query(getattr(plan, "web_search_query", None)),
    )


def _maybe_add_web_search(
    plan: DataRequirementsPlan, ctx: RequestContext, allowed: List[str]
) -> DataRequirementsPlan:
    if "web_search" not in allowed:
        return plan
    if "web_search" in plan.required_data or "web_search" in plan.optional_data:
        return plan
    if ctx.needs_recency or ctx.intent in {"education_q", "news_q", "single_stock_q", "sec_q"}:
        plan.optional_data.append("web_search")
    return plan


def _strip_ticker_dependent_data(plan: DataRequirementsPlan, ctx: RequestContext) -> DataRequirementsPlan:
    if ctx.tickers:
        return plan
    required = [item for item in plan.required_data if item not in {"fundamentals", "sec_snippets"}]
    optional = [item for item in plan.optional_data if item not in {"fundamentals", "sec_snippets"}]
    return DataRequirementsPlan(
        required_data=required,
        optional_data=optional,
        sec_sections=plan.sec_sections,
        notes=plan.notes,
    )

def _enforce_portfolio_requirements(plan: DataRequirementsPlan, ctx: RequestContext, allowed: List[str]) -> DataRequirementsPlan:
    if ctx.intent != "portfolio_q":
        return plan
    required = list(plan.required_data)
    if "portfolio_context" in allowed and "portfolio_context" not in required:
        required.append("portfolio_context")

    # if no tickers, strip ticker-only tools
    optional = list(plan.optional_data)
    if not ctx.tickers:
        required = [x for x in required if x not in {"fundamentals", "sec_snippets"}]
        optional = [x for x in optional if x not in {"fundamentals", "sec_snippets"}]

    return DataRequirementsPlan(
        required_data=required,
        optional_data=optional,
        sec_sections=plan.sec_sections,
        notes=plan.notes,
        web_search_query=getattr(plan, "web_search_query", None),
    )


def _news_items(tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
    for result in tool_results:
        if result.source != "get_news" or not result.ok:
            continue
        if isinstance(result.data, dict):
            items = result.data.get("items")
            if isinstance(items, list):
                return items
    return []


def _web_search_items(tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for result in tool_results:
        if result.source != "get_web_search" or not result.ok:
            continue
        data = result.data
        if not isinstance(data, dict):
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "summary": item.get("summary"),
                }
            )
    return out


def _log_web_search_results(tool_results: List[ToolResult], state: Dict[str, Any]) -> None:
    for result in tool_results:
        if result.source != "get_web_search" or not result.ok:
            continue
        data = result.data or {}
        if not isinstance(data, dict):
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        trimmed = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            trimmed.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "summary": item.get("summary"),
                }
            )
        _log_json(
            {
                "event": "web_search_results",
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "query": data.get("query"),
                "items": trimmed,
            }
        )


def _recency_insufficient(needs_recency: bool, tool_results: List[ToolResult]) -> bool:
    if not needs_recency:
        return False
    items = _news_items(tool_results)
    return not items


def _missing_sources(tool_results: List[ToolResult]) -> List[str]:
    return [res.source for res in tool_results if not res.ok]

def _build_missing_note(missing: List[str], recency_insufficient: bool) -> str:
    notes = []
    if missing:
        notes.append(f"Unavailable data sources: {', '.join(sorted(set(missing)))}.")
    if recency_insufficient:
        notes.append("Recent news may be unavailable or outdated.")
    return " ".join(notes).strip()


def _answer_mentions_web(answer: str, web_items: List[Dict[str, Any]]) -> bool:
    text = (answer or "").lower()
    if not text:
        return False
    if "web context" in text or "external sources" in text or "sources" in text:
        return True
    for item in web_items:
        url = (item.get("url") or "").lower()
        if url and url in text:
            return True
    return False


def _append_web_context(answer: str, web_items: List[Dict[str, Any]]) -> str:
    if not web_items:
        return answer
    lines = ["Web context:"]
    for item in web_items[:3]:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        url = (item.get("url") or "").strip()
        parts = []
        if title:
            parts.append(title)
        if summary:
            parts.append(summary)
        if url:
            parts.append(url)
        if parts:
            lines.append("- " + " | ".join(parts))
    if len(lines) == 1:
        return answer
    return f"{answer}\n\n" + "\n".join(lines)


def _plan_payload(state: GraphState) -> Dict[str, Any]:
    ctx = state.request_context or RequestContext()
    plan = state.data_requirements
    return {
        "trace_id": state.trace_id,
        "turn_id": state.turn_id,
        "intent": ctx.intent,
        "tickers": ctx.tickers,
        "required_data": plan.required_data,
        "optional_data": plan.optional_data,
    }


class DecisioningGraph:
    def __init__(self) -> None:
        self._planner_prompt = _load_prompt(PLANNER_PROMPT_PATH)
        self._synthesis_prompt = _load_prompt(SYNTHESIS_PROMPT_PATH)
        self._synthesis_stream_prompt = _load_prompt(SYNTHESIS_STREAM_PROMPT_PATH)
        self._no_tools_synthesis_prompt = _load_prompt(NO_TOOLS_SYNTHESIS_PROMPT_PATH)
        self._router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0.0)
        self._planner_llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0.0)
        self._synthesis_llm = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0.2)
        self._synthesis_llm_stream = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0.2, streaming=True)
        self._graph = self._build_graph(full_graph=True)
        self._pre_synthesis_graph = self._build_graph(full_graph=False)
        self._tool_executor = ToolExecutor()
        self._router_prompt = _load_prompt(ROUTER_PROMPT_PATH)

    @property
    def graph(self):
        return self._graph

    @property
    def pre_synthesis_graph(self):
        return self._pre_synthesis_graph

    async def router(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "router", state)
        start = time.perf_counter()

        message = state.get("message") or ""
        memory = _as_memory(state.get("memory"))
        recent_turns = memory.recent_turns[-4:] if memory.recent_turns else []

        payload = {
            "message": message,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
            "recent_turns": recent_turns,
        }

        structured_llm = self._router_llm.with_structured_output(RouterOutput, method="function_calling")
        messages = [
            SystemMessage(content=self._router_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]

        token_usage = None
        try:
            out = await structured_llm.ainvoke(messages)
            token_usage = getattr(out, "usage_metadata", None)
        except Exception as exc:
            logger.warning("router failed: %s", exc)
            out = RouterOutput()

        # Normalize outputs
        ctx = _normalize_request_context(out.request_context)

        handled = bool(out.handled)
        answer = (out.answer or "").strip()
        route_mode = (out.route_mode or "light_tools")

        # Safety: handled must have an answer
        if handled and not answer:
            handled = False
            answer = ""

        # 🔒 Enforce quick-answer for pure education
        if (
            not handled
            and route_mode == "no_tools"
            and ctx.intent == "education_q"
            and not ctx.needs_portfolio
            and not ctx.needs_recency
        ):
            # If router gave an answer, treat as handled
            if answer:
                handled = True
            # If router did NOT give an answer, do NOT force handled
            # (otherwise user will get blank reply)
            # handled stays False

        # If handled, ensure mode is no_tools (clean logs)
        if handled:
            route_mode = "no_tools"

        _log_json(
            {
                "event": "llm_call",
                "node": "router",
                "model": ROUTER_MODEL,
                "prompt_version": "router_v1",
                "duration_ms": _elapsed_ms(start),
                "token_usage": token_usage,
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "handled": handled,
                "handled_reason": (out.handled_reason or "").strip(),
                "answer_len": len(answer),
                "route_mode": route_mode,
                "intent": ctx.intent,
                "needs_portfolio": ctx.needs_portfolio,
                "needs_recency": ctx.needs_recency,
            }
        )

        update: Dict[str, Any] = {
            "request_context": ctx,
            "handled": handled,
            "handled_reason": (out.handled_reason or "").strip(),
            "answer": answer,
            "debug": {
                "router_notes": out.notes,
                "route_mode": route_mode,
                "confidence": out.confidence,
            },
        }
        return update
    
    async def no_tools_synthesis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "no_tools_synthesis", state)
        start = time.perf_counter()

        ctx = _as_request_context(state.get("request_context"))
        memory = _as_memory(state.get("memory"))

        payload = {
            "message": state.get("message") or "",
            "request_context": ctx.model_dump(mode="json"),
            "user_profile": memory.user_profile,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
            "recent_turns": (memory.recent_turns or [])[-4:],
        }

        # Use same synthesis model, but different prompt (no tools)
        messages = [
            SystemMessage(content=self._no_tools_synthesis_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]

        token_usage = None
        try:
            res = await self._synthesis_llm.ainvoke(messages)
            token_usage = getattr(res, "usage_metadata", None)
            answer = (getattr(res, "content", None) or "").strip()
        except Exception as exc:
            logger.warning("no_tools_synthesis failed: %s", exc)
            answer = "Sorry — I couldn’t answer that right now."

        if not answer:
            answer = "Sorry — I couldn’t answer that right now."

        _log_json(
            {
                "event": "llm_call",
                "node": "no_tools_synthesis",
                "model": SYNTHESIS_MODEL,
                "prompt_version": "no_tools_synthesis_v1",
                "duration_ms": _elapsed_ms(start),
                "token_usage": token_usage,
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "intent": ctx.intent,
            }
        )

        return {"answer": answer}
    
    async def stream_no_tools_synthesis(self, state: GraphState):
        ctx = _as_request_context(state.request_context)
        memory = _as_memory(state.memory)

        # If router already produced an answer, just emit it
        if state.handled:
            answer = (state.answer or "").strip()
            if answer:
                yield answer
            return

        payload = {
            "message": state.message or "",
            "request_context": ctx.model_dump(mode="json"),
            "user_profile": memory.user_profile,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
            "recent_turns": (memory.recent_turns or [])[-4:],
        }

        messages = [
            SystemMessage(content=self._no_tools_synthesis_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]

        emitted = ""
        async for chunk in self._synthesis_llm_stream.astream(messages):
            text = (chunk.content or "")
            if not text:
                continue
            emitted += text
            yield text

        state.answer = emitted.strip() or "Sorry — I couldn’t answer that right now."


    async def load_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "load_memory", state)
        start = time.perf_counter()
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        memory = load_memory_snapshot(user_id, session_id, max_turns=CHAT_MAX_HISTORY)
        memory.user_profile = _load_user_profile(state.get("db"), user_id)
        recent_turns = memory.recent_turns or []
        last_roles = []
        for item in recent_turns[-2:]:
            if isinstance(item, dict):
                role = item.get("role")
                if isinstance(role, str):
                    last_roles.append(role)
        _log_json(
            {
                "event": "node_span",
                "node": "load_memory",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        _log_json(
            {
                "event": "memory_snapshot",
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "summary_len": len(memory.thread_summary or ""),
                "recent_entities_count": len(memory.recent_entities or []),
                "recent_turns_count": len(recent_turns),
                "last_roles": last_roles,
            }
        )
        return {"memory": memory}

    async def policy_and_budget(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "policy_and_budget", state)
        start = time.perf_counter()
        ctx = _as_request_context(state.get("request_context"))
        allowed = _allowed_data_types(ctx.intent)
        sec_sections_cap = 2
        if ctx.requested_sections:
            sec_sections_cap = max(sec_sections_cap, len(ctx.requested_sections))
        budgets = BudgetConfig(
            global_timeout_s=DEFAULT_GLOBAL_TIMEOUT_S,
            allowed_data_types=allowed,
            tool_budgets={
                "get_portfolio_context": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_items=8,  # used as top_n
                ),
                "get_news": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_results=5,
                ),
                "get_sec_snippets": ToolBudget(
                    max_calls=sec_sections_cap,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_sections=sec_sections_cap,
                    max_snippets=6,
                ),
                "get_fundamentals": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_symbols=5,
                ),
                "get_web_search": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_results=5,
                ),
            },
        )
        _log_json(
            {
                "event": "node_span",
                "node": "policy_and_budget",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"budgets": budgets}

    async def data_requirements_planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "data_requirements_planner", state)
        start = time.perf_counter()
        ctx = _as_request_context(state.get("request_context"))
        memory = _as_memory(state.get("memory"))
        payload = {
            "message": state.get("message") or "",
            "request_context": ctx.model_dump(mode="json"),
            "user_profile": memory.user_profile,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
        }
        structured_llm = self._planner_llm.with_structured_output(DataRequirementsPlan, method="function_calling")
        messages = [
            SystemMessage(content=self._planner_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]
        token_usage = None
        try:
            res = await structured_llm.ainvoke(messages)
            plan = res
            token_usage = getattr(res, "usage_metadata", None)
        except Exception as exc:
            logger.warning("planner failed: %s", exc)
            plan = DataRequirementsPlan()
        allowed = _as_budgets(state.get("budgets")).allowed_data_types
        plan = _sanitize_data_requirements(plan, allowed)
        plan = _maybe_add_web_search(plan, ctx, allowed)
        plan = _strip_ticker_dependent_data(plan, ctx)
        plan = _enforce_portfolio_requirements(plan, ctx, allowed)
        _log_json(
            {
                "event": "llm_call",
                "node": "data_requirements_planner",
                "model": PLANNER_MODEL,
                "prompt_version": "planner_v1",
                "duration_ms": _elapsed_ms(start),
                "token_usage": token_usage,
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"data_requirements": plan}

    async def tool_exec_parallel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "tool_exec_parallel", state)
        start = time.perf_counter()

        ctx = _as_request_context(state.get("request_context"))
        data_plan = _as_plan(state.get("data_requirements"))
        budgets = _as_budgets(state.get("budgets"))
        message = state.get("message") or ""

        # Normalize + filter allowed
        data_types = _normalize_list(data_plan.required_data + data_plan.optional_data)
        allowed = budgets.allowed_data_types
        data_types = [dtype for dtype in data_types if dtype in allowed]

        calls: List[ToolCallSpec] = []
        pre_results: List[ToolResult] = []
        pre_statuses: List[Dict[str, Any]] = []

        # We'll use this to feed get_news when user didn't specify tickers (portfolio_q)
        portfolio_context_result: Optional[ToolResult] = None

        def _register_tool_error(tool_name: str, error_type: str, msg: str) -> None:
            pre_results.append(
                ToolResult(
                    ok=False,
                    source=tool_name,
                    as_of=None,
                    latency_ms=0,
                    warnings=[],
                    data=None,
                    error=ToolError(type=error_type, message=msg, retryable=False),
                )
            )
            pre_statuses.append(
                {
                    "name": tool_name,
                    "status": "error",
                    "latency_ms": 0,
                    "error_type": error_type,
                }
            )

        # Common tool context (shared)
        tool_ctx = ToolContext(
            db=state.get("db"),
            finnhub=state.get("finnhub"),
            user_id=state.get("user_id"),
            user_currency=state.get("user_currency") or "USD",
            message=message,
            symbols=ctx.tickers,
            history=_as_memory(state.get("memory")).recent_turns,
            holdings_snapshot=None,
        )

        # ---------------------------------------------------------------------
        # Pass 1 (optional): execute portfolio_context FIRST so we can derive
        # symbols_for_news for portfolio questions without tickers.
        # ---------------------------------------------------------------------
        if "portfolio_context" in data_types:
            tool_name = DATA_TYPE_TO_TOOL.get("portfolio_context")
            if tool_name:
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                first_call = [
                    ToolCallSpec(
                        name=tool_name,
                        arguments={
                            "top_n": budget.max_items or 8,
                            "max_holdings": 25,
                        },
                        data_type="portfolio_context",
                    )
                ]
                first_results, first_statuses = await self._tool_executor.execute(
                    first_call,
                    tool_ctx,
                    budgets.tool_budgets,
                    budgets.global_timeout_s,
                )
                if first_results:
                    portfolio_context_result = first_results[0]
                    pre_results.extend(first_results)
                    pre_statuses.extend(first_statuses)

            # Remove so we don't call it again in the parallel pass
            data_types = [d for d in data_types if d != "portfolio_context"]

        # ---------------------------------------------------------------------
        # Pass 2: build remaining calls (parallel)
        # ---------------------------------------------------------------------
        for dtype in data_types:
            tool_name = DATA_TYPE_TO_TOOL.get(dtype)
            if not tool_name:
                continue

            elif dtype == "fundamentals":
                if not ctx.tickers:
                    _register_tool_error(tool_name, "missing_ticker", "No tickers provided for fundamentals.")
                    continue
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={"symbols": ctx.tickers},
                        data_type=dtype,
                    )
                )

            elif dtype == "news":
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())

                # Preferred symbols:
                # 1) explicit tickers from router
                # 2) portfolio_context.symbols_for_news (if portfolio_q)
                # 3) memory recent_entities ticker-ish fallback
                symbols: List[str] = list(ctx.tickers or [])

                if (not symbols) and ctx.intent == "portfolio_q" and portfolio_context_result and portfolio_context_result.ok:
                    data = portfolio_context_result.data or {}
                    if isinstance(data, dict):
                        s = data.get("symbols_for_news") or []
                        if isinstance(s, list):
                            symbols = [str(x).strip().upper() for x in s if x]

                if (not symbols) and ctx.intent == "portfolio_q":
                    mem = _as_memory(state.get("memory"))
                    symbols = [
                        e for e in (mem.recent_entities or [])
                        if isinstance(e, str) and e.isupper() and 1 <= len(e) <= 6
                    ][:2]

                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={
                            "symbols": symbols or None,
                            "topic": "markets" if not symbols else None,
                            "max_results": budget.max_results or 5,
                            "days": 7,
                        },
                        data_type=dtype,
                    )
                )

            elif dtype == "web_search":
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                query = data_plan.web_search_query
                if not query:
                    prefix = " ".join(ctx.tickers) + " " if ctx.tickers else ""
                    query = (prefix + message).strip()

                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={"query": query, "max_results": budget.max_results or 5},
                        data_type=dtype,
                    )
                )

            elif dtype == "sec_snippets":
                sections = data_plan.sec_sections or ctx.requested_sections or ["general"]
                sections = [s for s in sections if s in SEC_SECTIONS] or ["general"]

                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                max_sections = budget.max_sections or 2

                if not ctx.tickers:
                    _register_tool_error(tool_name, "missing_ticker", "No tickers provided for SEC snippets.")
                    continue

                for idx, section in enumerate(sections):
                    if idx >= max_sections:
                        _register_tool_error(tool_name, "cap_exceeded", "SEC section cap exceeded.")
                        continue
                    calls.append(
                        ToolCallSpec(
                            name=tool_name,
                            arguments={"symbol": ctx.tickers[0], "section": section},
                            data_type=dtype,
                        )
                    )

        # Execute remaining calls in parallel
        results: List[ToolResult] = []
        statuses: List[Dict[str, Any]] = []
        if calls:
            results, statuses = await self._tool_executor.execute(
                calls,
                tool_ctx,
                budgets.tool_budgets,
                budgets.global_timeout_s,
            )

        # Merge pre results/statuses (portfolio_context + any early errors)
        if pre_results:
            results = pre_results + (results or [])
        if pre_statuses:
            statuses = pre_statuses + (statuses or [])

        _log_web_search_results(results, state)

        for res in results:
            _log_json(
                {
                    "event": "tool_result",
                    "tool": res.source,
                    "ok": res.ok,
                    "latency_ms": res.latency_ms,
                    "error_type": res.error.type if res.error else None,
                    "trace_id": state.get("trace_id"),
                    "turn_id": state.get("turn_id"),
                }
            )

        _log_json(
            {
                "event": "node_span",
                "node": "tool_exec_parallel",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )

        return {"tool_results": results, "tool_statuses": statuses}

    async def recency_guard(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "recency_guard", state)
        start = time.perf_counter()
        ctx = _as_request_context(state.get("request_context"))
        tool_results = _as_tool_results(state.get("tool_results"))
        recency_insufficient = _recency_insufficient(ctx.needs_recency, tool_results)
        _log_json(
            {
                "event": "node_span",
                "node": "recency_guard",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"recency_insufficient": recency_insufficient}

    async def synthesis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "synthesis", state)
        start = time.perf_counter()
        ctx = _as_request_context(state.get("request_context"))
        memory = _as_memory(state.get("memory"))
        tool_results = _as_tool_results(state.get("tool_results"))
        missing_sources = _missing_sources(tool_results)
        if tool_results and not any(res.ok for res in tool_results):
            note = _build_missing_note(missing_sources, state.get("recency_insufficient", False))
            answer = "I couldn't retrieve the requested data right now."
            if note:
                answer = f"{answer} {note}"
            return {"answer": answer}
        payload = {
            "message": state.get("message") or "",
            "request_context": ctx.model_dump(mode="json"),
            "user_profile": memory.user_profile,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
            "recency_insufficient": state.get("recency_insufficient", False),
            "tool_results": [tr.model_dump(mode="json") for tr in tool_results],
            "web_search_items": _web_search_items(tool_results),
        }
        structured_llm = self._synthesis_llm.with_structured_output(SynthesisOutput, method="function_calling")
        messages = [
            SystemMessage(content=self._synthesis_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]
        token_usage = None
        try:
            res = await structured_llm.ainvoke(messages)
            answer = (res.answer or "").strip()
            token_usage = getattr(res, "usage_metadata", None)
        except Exception as exc:
            logger.warning("synthesis failed: %s", exc)
            answer = "Sorry, I couldn't complete that request right now."
        removed = False
        if "get_fundamentals" in missing_sources:
            answer = answer.strip()
            answer = f"{answer}\n\nNote: I couldn’t load fundamentals right now, so this is based on price/news/portfolio only."
        limitation_note = _build_missing_note(missing_sources, state.get("recency_insufficient", False))
        if limitation_note:
            answer = f"{answer}\n\nData limitations: {limitation_note}"
        if removed and "get_fundamentals" in missing_sources:
            answer = f"{answer}\n\nFundamental metrics are not available at the moment."
        web_items = _web_search_items(tool_results)
        if web_items and not _answer_mentions_web(answer, web_items):
            answer = _append_web_context(answer, web_items)
        _log_json(
            {
                "event": "llm_call",
                "node": "synthesis",
                "model": SYNTHESIS_MODEL,
                "prompt_version": "synthesis_v1",
                "duration_ms": _elapsed_ms(start),
                "token_usage": token_usage,
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"answer": answer}

    async def stream_synthesis(self, state: GraphState):
        ctx = _as_request_context(state.request_context)
        memory = _as_memory(state.memory)
        tool_results = _as_tool_results(state.tool_results)
        missing_sources = _missing_sources(tool_results)

        if state.handled:
            answer = (state.answer or "").strip()
            if answer:
                yield answer
            return

        if tool_results and not any(res.ok for res in tool_results):
            note = _build_missing_note(missing_sources, state.recency_insufficient)
            answer = "I couldn't retrieve the requested data right now."
            if note:
                answer = f"{answer} {note}"
            state.answer = answer
            yield answer
            return

        payload = {
            "message": state.message or "",
            "request_context": ctx.model_dump(mode="json"),
            "user_profile": memory.user_profile,
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
            "recency_insufficient": state.recency_insufficient,
            "tool_results": [tr.model_dump(mode="json") for tr in tool_results],
            "web_search_items": _web_search_items(tool_results),
        }

        messages = [
            SystemMessage(content=self._synthesis_stream_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]

        emitted = ""
        async for chunk in self._synthesis_llm_stream.astream(messages):
            text = (chunk.content or "")
            if not text:
                continue
            emitted += text
            yield text

        # Optional: add tool failure note if you want (usually model handles it already)
        limitation_note = _build_missing_note(missing_sources, state.recency_insufficient)
        if limitation_note and "Data limitations:" not in emitted:
            extra = f"\n\nData limitations: {limitation_note}"
            emitted += extra
            yield extra

        state.answer = emitted.strip()


    async def postprocess_and_store(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "postprocess_and_store", state)
        start = time.perf_counter()
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        message = state.get("message") or ""
        answer = state.get("answer") or ""
        ctx = _as_request_context(state.get("request_context"))
        memory = _as_memory(state.get("memory"))

        new_summary = update_thread_summary(memory.thread_summary, message, answer)
        save_thread_summary(user_id, session_id, new_summary)

        entities = _normalize_list(memory.recent_entities + ctx.tickers)
        if len(entities) > CHAT_ENTITIES_MAX:
            entities = entities[-CHAT_ENTITIES_MAX:]
        save_recent_entities(user_id, session_id, entities)

        append_chat_history(
            user_id,
            session_id,
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
        )

        _log_json(
            {
                "event": "eval_artifacts",
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "request_context": ctx.model_dump(mode="json"),
                "data_requirements": _as_plan(state.get("data_requirements")).model_dump(mode="json"),
                "tool_summary": [
                    {
                        "source": res.source,
                        "ok": res.ok,
                        "error_type": res.error.type if res.error else None,
                    }
                    for res in _as_tool_results(state.get("tool_results"))
                ],
                "answer_chars": len(answer),
            }
        )

        _log_json(
            {
                "event": "node_span",
                "node": "postprocess_and_store",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"memory": memory, "answer": answer}

    def _build_graph(self, full_graph: bool) -> Any:
        workflow: StateGraph = StateGraph(GraphStateDict)

        # Nodes
        workflow.add_node("load_memory", self.load_memory)
        workflow.add_node("router", self.router)
        workflow.add_node("policy_and_budget", self.policy_and_budget)
        workflow.add_node("data_requirements_planner", self.data_requirements_planner)
        workflow.add_node("tool_exec_parallel", self.tool_exec_parallel)
        workflow.add_node("recency_guard", self.recency_guard)
        workflow.add_node("synthesis", self.synthesis)
        workflow.add_node("no_tools_synthesis", self.no_tools_synthesis)
        workflow.add_node("postprocess_and_store", self.postprocess_and_store)

        # Entry
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "router")

        def route_key(state: Dict[str, Any]) -> str:
            # If we already have an answer, end early
            if (state.get("answer") or "").strip():
                return "handled"
            if state.get("handled"):
                return "handled"
            if state.get("debug", {}).get("route_mode") == "no_tools":
                return "no_tools"
            return "continue"

        if full_graph:
            router_targets = {
                "handled": "postprocess_and_store",
                "no_tools": "no_tools_synthesis",
                "continue": "policy_and_budget",
            }
        else:
            # Pre-synthesis graph should not store; it ends after synthesis/no-tools
            router_targets = {
                "handled": END,
                "no_tools": END,
                "continue": "policy_and_budget",
            }

        workflow.add_conditional_edges("router", route_key, router_targets)

        # Tool path
        workflow.add_edge("policy_and_budget", "data_requirements_planner")
        workflow.add_edge("data_requirements_planner", "tool_exec_parallel")
        workflow.add_edge("tool_exec_parallel", "recency_guard")
        workflow.add_edge("recency_guard", "synthesis")

        # Final edges
        if full_graph:
            workflow.add_edge("synthesis", "postprocess_and_store")
            workflow.add_edge("no_tools_synthesis", "postprocess_and_store")
            workflow.add_edge("postprocess_and_store", END)
        else:
            workflow.add_edge("synthesis", END)
            workflow.add_edge("no_tools_synthesis", END)

        return workflow.compile()


    async def run(self, state: GraphState) -> GraphState:
        payload = state.model_dump()
        _log_json(
            {
                "event": "graph_start",
                "trace_id": payload.get("trace_id"),
                "turn_id": payload.get("turn_id"),
            }
        )
        final_state = await self._graph.ainvoke(payload)
        merged = {**payload, **final_state}
        _log_json(
            {
                "event": "graph_end",
                "trace_id": payload.get("trace_id"),
                "turn_id": payload.get("turn_id"),
            }
        )
        return GraphState.model_validate(merged)

    async def run_pre_synthesis(self, state: GraphState) -> GraphState:
        payload = state.model_dump()
        _log_json(
            {
                "event": "graph_start",
                "trace_id": payload.get("trace_id"),
                "turn_id": payload.get("turn_id"),
                "mode": "pre_synthesis",
            }
        )
        final_state = await self._pre_synthesis_graph.ainvoke(payload)
        merged = {**payload, **final_state}
        _log_json(
            {
                "event": "graph_end",
                "trace_id": payload.get("trace_id"),
                "turn_id": payload.get("turn_id"),
                "mode": "pre_synthesis",
            }
        )
        return GraphState.model_validate(merged)

    async def run_synthesis(self, state: GraphState) -> GraphState:
        if state.handled:
            return state

        route_mode = (state.debug or {}).get("route_mode")
        if route_mode == "no_tools":
            update = await self.no_tools_synthesis(state.model_dump())
        else:
            update = await self.synthesis(state.model_dump())

        state.answer = update.get("answer") or ""
        return state

    async def run_postprocess(self, state: GraphState) -> GraphState:
        await self.postprocess_and_store(state.model_dump())
        return state

    @staticmethod
    def new_state(
        *,
        message: str,
        user_id: Any,
        user_currency: str,
        session_id: str,
        db: Any,
        finnhub: Any,
    ) -> GraphState:
        return GraphState(
            message=message,
            user_id=user_id,
            user_currency=user_currency,
            session_id=session_id,
            trace_id=uuid.uuid4().hex,
            turn_id=uuid.uuid4().hex,
            db=db,
            finnhub=finnhub,
        )

    @staticmethod
    def build_plan_event(state: GraphState) -> Dict[str, Any]:
        return _plan_payload(state)
