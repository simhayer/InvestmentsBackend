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
    IntentRefinementOutput,
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
INTENT_PROMPT_PATH = PROMPT_DIR / "intent_v1.txt"
PLANNER_PROMPT_PATH = PROMPT_DIR / "planner_v1.txt"
SYNTHESIS_PROMPT_PATH = PROMPT_DIR / "synthesis_v1.txt"

INTENT_MODEL = os.getenv("OPENAI_INTENT_MODEL", os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini"))
PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
SYNTHESIS_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

DEFAULT_TOOL_TIMEOUT_S = float(os.getenv("CHAT_TOOL_TIMEOUT_SEC", "6"))
DEFAULT_GLOBAL_TIMEOUT_S = float(os.getenv("CHAT_TOOL_GLOBAL_TIMEOUT_SEC", "12"))

DATA_TYPE_TO_TOOL = {
    "portfolio_summary": "get_portfolio_summary",
    "holdings": "get_holdings",
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


def _merge_request_context(base: RequestContext, refined: IntentRefinementOutput) -> RequestContext:
    intent = refined.intent or base.intent
    needs_portfolio = refined.needs_portfolio if refined.needs_portfolio is not None else base.needs_portfolio
    needs_recency = refined.needs_recency if refined.needs_recency is not None else base.needs_recency
    if needs_portfolio and intent != "portfolio_q":
        intent = "portfolio_q"
    sections = _normalize_list((base.requested_sections or []) + (refined.requested_sections or []))
    output_style = refined.output_style or base.output_style
    risk_flags = set(base.risk_flags)
    if refined.risk_flags:
        risk_flags.update(refined.risk_flags)
    return RequestContext(
        intent=intent,
        tickers=base.tickers,
        needs_portfolio=needs_portfolio,
        needs_recency=needs_recency,
        requested_sections=sections or None,
        output_style=output_style,
        risk_flags=risk_flags,
        timeframe=base.timeframe,
        user_constraints=base.user_constraints,
    )


def parse_request_context(message: str) -> RequestContext:
    text = (message or "").strip()
    lowered = text.lower()

    tickers: List[str] = []
    for match in re.findall(r"\$([A-Za-z]{1,5})", text):
        tickers.append(match.upper())
    for match in re.findall(r"\(([A-Za-z]{1,5})\)", text):
        tickers.append(match.upper())
    for match in re.findall(r"\bticker\s+([A-Za-z]{1,5})\b", text, flags=re.IGNORECASE):
        tickers.append(match.upper())
    for match in re.findall(r"\b[A-Z]{2,5}\b", text):
        if match in {"USD", "SEC", "CPI", "GDP", "CEO", "CFO"}:
            continue
        tickers.append(match.upper())
    tickers = _normalize_list(tickers)

    portfolio_terms = r"(portfolio|holdings?|positions?|stocks?|shares|investments?|account)"
    possession_terms = r"(own|have|hold)"
    needs_portfolio = any(
        phrase in lowered
        for phrase in [
            "portfolio",
            "holdings",
            "positions",
            "my account",
            "my investments",
        ]
    )
    if not needs_portfolio:
        if re.search(rf"\bmy\s+{portfolio_terms}\b", lowered):
            needs_portfolio = True
        elif re.search(rf"\bwhat\s+{portfolio_terms}\b.*\b(i|my)\b", lowered):
            needs_portfolio = True
        elif re.search(rf"\b(i|my)\b.*\b{portfolio_terms}\b.*\b{possession_terms}\b", lowered):
            needs_portfolio = True
        elif re.search(rf"\b{portfolio_terms}\b.*\b{possession_terms}\b", lowered) and "stock" in lowered:
            needs_portfolio = True
        elif re.search(r"\b(list|show|see)\b.*\b(my\s+)?(holdings?|positions?|stocks?|shares)\b", lowered):
            needs_portfolio = True

    needs_recency = any(
        phrase in lowered
        for phrase in ["latest", "today", "recent", "news", "breaking", "this week", "now"]
    )

    requested_sections: List[str] = []
    if "risk" in lowered:
        requested_sections.append("risk")
    if "mda" in lowered or "md&a" in lowered or "management discussion" in lowered:
        requested_sections.append("mda")
    if "business" in lowered:
        requested_sections.append("business")
    if "overview" in lowered or "general" in lowered:
        requested_sections.append("general")
    requested_sections = _normalize_list(requested_sections)

    output_style = "short"
    if any(word in lowered for word in ["detailed", "detail", "deep dive", "long"]):
        output_style = "long"
    if any(word in lowered for word in ["brief", "short", "quick"]):
        output_style = "short"

    risk_flags = set()
    if "sell everything" in lowered or "panic" in lowered:
        risk_flags.add("panic_sell")
    if "day trade" in lowered or "day-trade" in lowered:
        risk_flags.add("day_trading")
    if "options" in lowered or "calls" in lowered or "puts" in lowered:
        risk_flags.add("options")

    timeframe = None
    match = re.search(r"\b(last|past)\s+(\d+)\s+(day|week|month|year)s?\b", lowered)
    if match:
        timeframe = f"{match.group(1)} {match.group(2)} {match.group(3)}s"
    if "ytd" in lowered or "year to date" in lowered:
        timeframe = "year to date"

    user_constraints = []
    if "no crypto" in lowered or "avoid crypto" in lowered:
        user_constraints.append("no_crypto")
    if "no options" in lowered or "avoid options" in lowered:
        user_constraints.append("no_options")

    intent: IntentType = "education_q"
    if needs_portfolio:
        intent = "portfolio_q"
    elif any(term in lowered for term in ["sec", "10-k", "10q", "10-q", "filing"]):
        intent = "sec_q"
    elif any(term in lowered for term in ["news", "headline", "latest", "today"]):
        intent = "news_q"
    elif any(term in lowered for term in ["who are you", "what can you do", "how do you work", "help"]):
        intent = "meta_q"
    elif tickers:
        intent = "single_stock_q"

    return RequestContext(
        intent=intent,
        tickers=tickers,
        needs_portfolio=needs_portfolio,
        needs_recency=needs_recency,
        requested_sections=requested_sections or None,
        output_style=output_style,
        risk_flags=risk_flags,
        timeframe=timeframe,
        user_constraints=user_constraints,
    )


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
        return ["portfolio_summary", "holdings", "fundamentals", "news", "sec_snippets"]
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


def _enforce_portfolio_requirements(
    plan: DataRequirementsPlan, ctx: RequestContext, allowed: List[str]
) -> DataRequirementsPlan:
    if ctx.intent != "portfolio_q":
        return plan
    required = list(plan.required_data)
    for dtype in ("portfolio_summary", "holdings"):
        if dtype in allowed and dtype not in required:
            required.append(dtype)
    optional = list(plan.optional_data)
    if not ctx.tickers:
        required = [item for item in required if item not in {"fundamentals", "sec_snippets"}]
        optional = [item for item in optional if item not in {"fundamentals", "sec_snippets"}]
    return DataRequirementsPlan(
        required_data=required,
        optional_data=optional,
        sec_sections=plan.sec_sections,
        notes=plan.notes,
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


def _strip_sentences(text: str, banned_terms: tuple[str, ...]) -> tuple[str, bool]:
    if not text:
        return "", False
    sentences = re.split(r"(?<=[.!?])\\s+", text.strip())
    kept: List[str] = []
    removed = False
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in banned_terms):
            removed = True
            continue
        kept.append(sentence)
    return " ".join(kept).strip(), removed


def _ensure_disclaimer(text: str) -> str:
    if "financial advice" in (text or "").lower():
        return text
    if not text:
        return "This is not financial advice."
    return f"{text}\n\nThis is not financial advice."


def _build_missing_note(missing: List[str], recency_insufficient: bool) -> str:
    notes = []
    if missing:
        notes.append(f"Unavailable data sources: {', '.join(sorted(set(missing)))}.")
    if recency_insufficient:
        notes.append("Recent news may be unavailable or stale.")
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


def evaluate_recency(needs_recency: bool, tool_results: List[ToolResult]) -> bool:
    return _recency_insufficient(needs_recency, tool_results)


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
        self._intent_prompt = _load_prompt(INTENT_PROMPT_PATH)
        self._planner_prompt = _load_prompt(PLANNER_PROMPT_PATH)
        self._synthesis_prompt = _load_prompt(SYNTHESIS_PROMPT_PATH)
        self._intent_llm = ChatOpenAI(model=INTENT_MODEL, temperature=0.0)
        self._planner_llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0.0)
        self._synthesis_llm = ChatOpenAI(model=SYNTHESIS_MODEL, temperature=0.2)
        self._graph = self._build_graph(full_graph=True)
        self._pre_synthesis_graph = self._build_graph(full_graph=False)
        self._tool_executor = ToolExecutor()

    @property
    def graph(self):
        return self._graph

    @property
    def pre_synthesis_graph(self):
        return self._pre_synthesis_graph

    async def ingest_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "ingest_request", state)
        start = time.perf_counter()
        ctx = parse_request_context(state.get("message") or "")
        _log_json(
            {
                "event": "node_span",
                "node": "ingest_request",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"request_context": ctx}

    async def load_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "load_memory", state)
        start = time.perf_counter()
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        memory = load_memory_snapshot(user_id, session_id, max_turns=CHAT_MAX_HISTORY)
        memory.user_profile = _load_user_profile(state.get("db"), user_id)
        _log_json(
            {
                "event": "node_span",
                "node": "load_memory",
                "duration_ms": _elapsed_ms(start),
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"memory": memory}

    async def intent_refinement(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log_step("node_start", "intent_refinement", state)
        start = time.perf_counter()
        ctx = _as_request_context(state.get("request_context"))
        memory = _as_memory(state.get("memory"))
        payload = {
            "message": state.get("message") or "",
            "request_context": ctx.model_dump(mode="json"),
            "thread_summary": memory.thread_summary,
            "recent_entities": memory.recent_entities,
        }
        structured_llm = self._intent_llm.with_structured_output(
            IntentRefinementOutput, method="function_calling"
        )
        messages = [
            SystemMessage(content=self._intent_prompt),
            HumanMessage(content=json.dumps(payload, separators=(",", ":"))),
        ]
        token_usage = None
        try:
            res = await structured_llm.ainvoke(messages)
            token_usage = getattr(res, "usage_metadata", None)
            merged = _merge_request_context(ctx, res)
        except Exception as exc:
            logger.warning("intent_refinement failed: %s", exc)
            merged = ctx
        _log_json(
            {
                "event": "intent_refined",
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
                "intent": merged.intent,
                "needs_portfolio": merged.needs_portfolio,
                "needs_recency": merged.needs_recency,
            }
        )
        _log_json(
            {
                "event": "llm_call",
                "node": "intent_refinement",
                "model": INTENT_MODEL,
                "prompt_version": "intent_v1",
                "duration_ms": _elapsed_ms(start),
                "token_usage": token_usage,
                "trace_id": state.get("trace_id"),
                "turn_id": state.get("turn_id"),
            }
        )
        return {"request_context": merged}

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
                "get_holdings": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_items=25,
                ),
                "get_portfolio_summary": ToolBudget(
                    max_calls=1,
                    timeout_s=DEFAULT_TOOL_TIMEOUT_S,
                    max_items=5,
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

        data_types = _normalize_list(data_plan.required_data + data_plan.optional_data)
        allowed = budgets.allowed_data_types
        data_types = [dtype for dtype in data_types if dtype in allowed]

        calls: List[ToolCallSpec] = []
        pre_results: List[ToolResult] = []
        pre_statuses: List[Dict[str, Any]] = []
        for dtype in data_types:
            tool_name = DATA_TYPE_TO_TOOL.get(dtype)
            if not tool_name:
                continue
            if dtype == "portfolio_summary":
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={"top_n": budget.max_items or 5},
                        data_type=dtype,
                    )
                )
            elif dtype == "holdings":
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={"max_items": budget.max_items or 25},
                        data_type=dtype,
                    )
                )
            elif dtype == "fundamentals":
                if not ctx.tickers:
                    pre_results.append(
                        ToolResult(
                            ok=False,
                            source=tool_name,
                            as_of=None,
                            latency_ms=0,
                            warnings=[],
                            data=None,
                            error=ToolError(
                                type="missing_ticker",
                                message="No tickers provided for fundamentals.",
                                retryable=False,
                            ),
                        )
                    )
                    pre_statuses.append(
                        {
                            "name": tool_name,
                            "status": "error",
                            "latency_ms": 0,
                            "error_type": "missing_ticker",
                        }
                    )
                    continue
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={"symbols": ctx.tickers},
                        data_type=dtype,
                    )
                )
            elif dtype == "news":
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={},
                        data_type=dtype,
                    )
                )
            elif dtype == "web_search":
                calls.append(
                    ToolCallSpec(
                        name=tool_name,
                        arguments={},
                        data_type=dtype,
                    )
                )
            elif dtype == "sec_snippets":
                sections = data_plan.sec_sections or ctx.requested_sections or ["general"]
                sections = [s for s in sections if s in SEC_SECTIONS]
                if not sections:
                    sections = ["general"]
                budget = budgets.tool_budgets.get(tool_name, ToolBudget())
                max_sections = budget.max_sections or 2
                if not ctx.tickers:
                    pre_results.append(
                        ToolResult(
                            ok=False,
                            source=tool_name,
                            as_of=None,
                            latency_ms=0,
                            warnings=[],
                            data=None,
                            error=ToolError(
                                type="missing_ticker",
                                message="No tickers provided for SEC snippets.",
                                retryable=False,
                            ),
                        )
                    )
                    pre_statuses.append(
                        {
                            "name": tool_name,
                            "status": "error",
                            "latency_ms": 0,
                            "error_type": "missing_ticker",
                        }
                    )
                    continue
                for idx, section in enumerate(sections):
                    if idx >= max_sections:
                        pre_results.append(
                            ToolResult(
                                ok=False,
                                source=tool_name,
                                as_of=None,
                                latency_ms=0,
                                warnings=[],
                                data=None,
                                error=ToolError(
                                    type="cap_exceeded",
                                    message="SEC section cap exceeded.",
                                    retryable=False,
                                ),
                            )
                        )
                        pre_statuses.append(
                            {
                                "name": tool_name,
                                "status": "error",
                                "latency_ms": 0,
                                "error_type": "cap_exceeded",
                            }
                        )
                        continue
                    calls.append(
                        ToolCallSpec(
                            name=tool_name,
                            arguments={"symbol": (ctx.tickers[0] if ctx.tickers else ""), "section": section},
                            data_type=dtype,
                        )
                    )

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

        results, statuses = await self._tool_executor.execute(
            calls,
            tool_ctx,
            budgets.tool_budgets,
            budgets.global_timeout_s,
        )
        if pre_results:
            results = pre_results + results
        if pre_statuses:
            statuses = pre_statuses + statuses

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
            return {"answer": _ensure_disclaimer(answer)}
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
            answer, removed = _strip_sentences(answer, FUNDAMENTAL_TERMS)
        if not answer:
            answer = "I couldn't retrieve the requested data right now."
        limitation_note = _build_missing_note(missing_sources, state.get("recency_insufficient", False))
        if limitation_note:
            answer = f"{answer}\n\nData limitations: {limitation_note}"
        if removed and "get_fundamentals" in missing_sources:
            answer = f"{answer}\n\nFundamental metrics are not available at the moment."
        web_items = _web_search_items(tool_results)
        if web_items and not _answer_mentions_web(answer, web_items):
            answer = _append_web_context(answer, web_items)
        answer = _ensure_disclaimer(answer)
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
        workflow.add_node("ingest_request", self.ingest_request)
        workflow.add_node("load_memory", self.load_memory)
        workflow.add_node("intent_refinement", self.intent_refinement)
        workflow.add_node("policy_and_budget", self.policy_and_budget)
        workflow.add_node("data_requirements_planner", self.data_requirements_planner)
        workflow.add_node("tool_exec_parallel", self.tool_exec_parallel)
        workflow.add_node("recency_guard", self.recency_guard)
        if full_graph:
            workflow.add_node("synthesis", self.synthesis)
            workflow.add_node("postprocess_and_store", self.postprocess_and_store)

        workflow.set_entry_point("ingest_request")
        workflow.add_edge("ingest_request", "load_memory")
        workflow.add_edge("load_memory", "intent_refinement")
        workflow.add_edge("intent_refinement", "policy_and_budget")
        workflow.add_edge("policy_and_budget", "data_requirements_planner")
        workflow.add_edge("data_requirements_planner", "tool_exec_parallel")
        workflow.add_edge("tool_exec_parallel", "recency_guard")
        if full_graph:
            workflow.add_edge("recency_guard", "synthesis")
            workflow.add_edge("synthesis", "postprocess_and_store")
            workflow.add_edge("postprocess_and_store", END)
        else:
            workflow.add_edge("recency_guard", END)

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
