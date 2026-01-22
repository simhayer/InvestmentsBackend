from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.cache.cache_backend import cache_get, cache_set
from services.holding_service import get_all_holdings
from services.ai.chat_agent.tools import (
    TOOL_REGISTRY,
    ToolContext,
    ToolSelection,
    prepare_tool_args,
    render_tool_manifest,
    validate_tool_args,
)
from .types import ChatState, IntentResult

logger = logging.getLogger("chat_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.2)
llm_stream = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.2,
    streaming=True,
)
planner_llm = ChatOpenAI(
    model=os.getenv("OPENAI_PLANNER_MODEL", "gpt-4.1-mini"),
    temperature=0.0,
)

CHAT_HISTORY_TTL_SEC = int(os.getenv("CHAT_HISTORY_TTL_SEC", "21600"))  # 6h
CHAT_MAX_HISTORY = int(os.getenv("CHAT_MAX_HISTORY", "14"))
MAX_SYMBOLS = int(os.getenv("CHAT_MAX_SYMBOLS", "3"))
MAX_HOLDINGS = int(os.getenv("CHAT_MAX_HOLDINGS", "25"))

_EXPLICIT_SYMBOL_RE = re.compile(r"\$([A-Za-z]{1,6})\b")
_HOLDINGS_Q_RE = re.compile(r"\b(what|which)\b.*\b(own|hold|holdings|positions)\b", re.IGNORECASE)
_OWN_RE = re.compile(r"\b(?:do i|do we|do you|i|we)\s+(?:own|have|hold)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\b(?:what|which|list|show)\b.*\b(holdings|positions|stocks|portfolio)\b", re.IGNORECASE)
_SYMBOL_AFTER_OWN_RE = re.compile(r"\b(?:own|have|hold)\s+([A-Za-z]{1,12})\b", re.IGNORECASE)
_STOPWORDS = {
    "any",
    "some",
    "stocks",
    "stock",
    "holdings",
    "positions",
    "portfolio",
    "shares",
    "etf",
    "etfs",
    "crypto",
    "cryptos",
    "already",
    "currently",
}


class ToolPlan(BaseModel):
    allowed_tools: List[str] = Field(default_factory=list)


def _history_key(user_id: Any, session_id: str) -> str:
    return f"CHAT:SESSION:{str(user_id)}:{(session_id or '').strip()}"


def load_chat_history(user_id: Any, session_id: str) -> List[Dict[str, str]]:
    key = _history_key(user_id, session_id)
    payload = cache_get(key)
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [m for m in messages if isinstance(m, dict)]
    return []


def save_chat_history(user_id: Any, session_id: str, messages: List[Dict[str, str]]) -> None:
    key = _history_key(user_id, session_id)
    cache_set(key, {"messages": messages}, ttl_seconds=CHAT_HISTORY_TTL_SEC)


def append_chat_history(
    user_id: Any,
    session_id: str,
    new_messages: List[Dict[str, str]],
    max_items: int = CHAT_MAX_HISTORY,
) -> List[Dict[str, str]]:
    history = load_chat_history(user_id, session_id)
    history.extend(new_messages)
    if len(history) > max_items:
        history = history[-max_items:]
    save_chat_history(user_id, session_id, history)
    return history

def _format_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for msg in history[-CHAT_MAX_HISTORY:]:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _explicit_symbols_from_text(text: str) -> List[str]:
    raw = _EXPLICIT_SYMBOL_RE.findall(text or "")
    return [s.upper() for s in raw if s]


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _build_answer_messages(
    state: ChatState,
    tool_results: List[Dict[str, Any]],
) -> List[Any]:
    intent = state.get("intent") or ""
    message = state.get("message") or ""
    history = state.get("history") or []
    symbols = state.get("symbols") or []

    history_block = _format_history(history)
    system_content = """
You are a senior financial analyst and stock expert.
Stay strictly within stocks, ETFs, crypto, portfolio, and market analysis.

Guardrails:
- Do NOT invent numbers. If data is missing, say "Not available".
- Use SEC snippets as authoritative for company fundamentals and risks.
- Use fundamentals data (if present) for concrete metrics only.
- If the user asks about a company but no explicit ticker is provided, ask for the ticker.
- Avoid trading instructions. Provide analysis and education only.
- If portfolio data is provided, only reference those holdings.
- Use only the tool outputs provided; do not hallucinate extra data.

Personalization:
- Tailor tone to the user's onboarding profile when present (risk level, time horizon, experience).
- Include only sections that have supporting data (skip empty sections).
- If news is provided, include a short "News" section grounded in those items.
- If SEC snippets are provided, include a short "SEC" section (business/risks/MD&A as available).
- If portfolio data is provided, include a "Portfolio" section with concrete holdings/summary.
- If fundamentals are provided, include key metrics and note gaps when relevant.

Response style:
- Be concise and structured.
- Use short paragraphs or bullets.
- End with a short prompt for follow-up if needed.
""".strip()

    user_content = f"""
CHAT HISTORY:
{history_block}

USER MESSAGE:
{message}

INTENT:
{intent}

SYMBOLS:
{symbols}
""".strip()

    tool_calls = []
    tool_messages: List[ToolMessage] = []
    for idx, item in enumerate(tool_results or []):
        name = item.get("name") or ""
        args = item.get("arguments") or {}
        data = item.get("data") or {}
        if not name:
            continue
        tool_call_id = f"tool_call_{idx}"
        tool_calls.append({"name": name, "args": args, "id": tool_call_id})
        content = json.dumps(_json_safe(data), separators=(",", ":"))
        tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    messages: List[Any] = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
    if tool_calls:
        messages.append(AIMessage(content="", tool_calls=tool_calls))
        messages.extend(tool_messages)
    return messages


def _is_holdings_list_question(text: str) -> bool:
    t = text or ""
    if _LIST_RE.search(t):
        return True
    t_low = t.lower()
    if "own" in t_low and ("stock" in t_low or "stocks" in t_low or "holdings" in t_low or "positions" in t_low):
        return True
    return False


def _extract_symbol_candidate(text: str) -> str | None:
    if not text:
        return None
    hit = _EXPLICIT_SYMBOL_RE.search(text)
    if hit:
        return hit.group(1).upper()
    hit = _SYMBOL_AFTER_OWN_RE.search(text)
    if hit:
        cand = hit.group(1).strip()
        if cand and cand.lower() not in _STOPWORDS:
            return cand.upper()
    caps = re.findall(r"\b[A-Z]{2,6}\b", text)
    for c in caps:
        if c.lower() not in _STOPWORDS:
            return c.upper()
    return None


def _aggregate_holdings(rows: List[Any]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sym = (getattr(row, "symbol", "") or "").strip().upper()
        if not sym:
            continue
        rec = out.get(sym)
        if not rec:
            rec = {"symbol": sym, "name": getattr(row, "name", None), "quantity": 0.0}
            out[sym] = rec
        qty = getattr(row, "quantity", None)
        try:
            rec["quantity"] += float(qty or 0.0)
        except (TypeError, ValueError):
            pass
    return list(out.values())


def _render_holdings_list(rows: List[Any]) -> str:
    holdings = _aggregate_holdings(rows)
    if not holdings:
        return "I couldn't find any holdings on your account."
    lines = ["Here are the holdings I can see:"]
    for h in holdings:
        sym = h.get("symbol") or "Unknown"
        name = h.get("name")
        qty = h.get("quantity")
        label = f"{sym}"
        if name:
            label = f"{sym} ({name})"
        if qty:
            lines.append(f"- {label}, qty {qty}")
        else:
            lines.append(f"- {label}")
    lines.append("Want a breakdown by performance or sector?")
    return "\n".join(lines)


def _render_ownership_answer(rows: List[Any], candidate: str) -> str:
    if not rows:
        return "I couldn't find any holdings on your account."
    sym = candidate.strip().upper()
    matched = []
    for row in rows:
        row_sym = (getattr(row, "symbol", "") or "").strip().upper()
        row_name = (getattr(row, "name", "") or "").strip().lower()
        if sym == row_sym or sym.lower() in row_name:
            matched.append(row)
    if matched:
        qty = sum(float(getattr(r, "quantity", 0.0) or 0.0) for r in matched)
        if qty:
            return f"Yes — you own {sym}. Total quantity: {qty}."
        return f"Yes — you own {sym}."
    return f"I don't see {sym} in your holdings. If you meant a ticker, try using the ticker symbol (e.g., $TSLA)."


async def fast_path_node(state: ChatState) -> Dict[str, Any]:
    message = (state.get("message") or "").strip()
    db: Session = state.get("db")
    user_id = state.get("user_id")

    if not message or db is None or user_id is None:
        return {"short_circuit": False}

    if _is_holdings_list_question(message):
        rows = get_all_holdings(str(user_id), db)
        return {
            "answer": _render_holdings_list(rows),
            "short_circuit": True,
            "fast_path_reason": "holdings_list",
            "debug": {"fast_path": "holdings_list"},
        }

    if _OWN_RE.search(message):
        candidate = _extract_symbol_candidate(message)
        if not candidate:
            return {"short_circuit": False}
        rows = get_all_holdings(str(user_id), db)
        return {
            "answer": _render_ownership_answer(rows, candidate),
            "short_circuit": True,
            "fast_path_reason": "holdings_check",
            "debug": {"fast_path": "holdings_check", "symbol": candidate},
        }

    return {"short_circuit": False}


def _render_holdings_answer(holdings: List[Dict[str, Any]]) -> str:
    if not holdings:
        return "I could not find any holdings on your account."
    lines = ["Here are the current holdings I can see:"]
    for h in holdings:
        sym = h.get("symbol") or "Unknown"
        name = h.get("name")
        qty = h.get("quantity")
        val = h.get("value")
        cur = h.get("currency") or "USD"
        label = f"{sym}"
        if name:
            label = f"{sym} ({name})"
        parts = [label]
        if qty is not None:
            parts.append(f"qty {qty}")
        if val is not None:
            parts.append(f"value {val} {cur}")
        lines.append("- " + ", ".join(parts))
    lines.append("Want a breakdown by sector or performance?")
    return "\n".join(lines)


async def route_node(state: ChatState) -> Dict[str, Any]:
    message = (state.get("message") or "").strip()
    history = state.get("history") or []
    history_block = _format_history(history)
    explicit = _explicit_symbols_from_text(message)

    structured_llm = llm.with_structured_output(IntentResult, method="function_calling")
    prompt = f"""
        You are a router for a finance assistant.

        Decide if the user request is related to stocks, ETFs, crypto, portfolio, market news, or investing education.
        If it is NOT related, set intent to "off_topic".

        Only include symbols that the user explicitly typed (tickers like AAPL, TSLA, or crypto like BTC).
        Do NOT guess tickers from company names.

        Provide:
        - intent: stock_analysis | portfolio | crypto | market_news | education | off_topic
        - symbols: list of explicit tickers/crypto symbols
        - needs_portfolio: true if the user references "my portfolio", "my holdings", or "my positions"
        - needs_user_profile: true if personalization would help

        CHAT HISTORY:
        {history_block}

        USER MESSAGE:
        {message}
        """

    res = await structured_llm.ainvoke(prompt)
    symbols = [s.strip().upper() for s in res.symbols if s and isinstance(s, str)]
    if explicit:
        symbols = list(dict.fromkeys(explicit + symbols))

    needs_portfolio = res.needs_portfolio or ("portfolio" in message.lower() or "holdings" in message.lower())

    return {
        "intent": res.intent,
        "symbols": symbols[:MAX_SYMBOLS],
        "needs_portfolio": needs_portfolio,
        "needs_user_profile": res.needs_user_profile,
    }


def _default_tool_caps(symbols: List[str]) -> Dict[str, Any]:
    sec_max_calls = 4 if symbols else 0
    return {
        "get_user_profile": {"max_calls": 1},
        "get_portfolio_summary": {"max_calls": 1, "top_n": 5},
        "get_holdings": {"max_calls": 1, "max_items": MAX_HOLDINGS},
        "get_fundamentals": {"max_calls": 1, "max_symbols": MAX_SYMBOLS},
        "get_sec_snippets": {"max_calls": sec_max_calls, "max_snippets": 6},
        "get_news": {"max_calls": 1, "max_results": 6},
    }


async def plan_node(state: ChatState) -> Dict[str, Any]:
    message = (state.get("message") or "").strip()
    intent = (state.get("intent") or "").strip()
    symbols = state.get("symbols") or []
    needs_portfolio = bool(state.get("needs_portfolio"))
    needs_user_profile = bool(state.get("needs_user_profile"))

    history_block = _format_history(state.get("history") or [])
    tool_manifest = render_tool_manifest(list(TOOL_REGISTRY.keys()))

    structured_llm = planner_llm.with_structured_output(ToolPlan, method="function_calling")
    prompt = f"""
        You are a planner that decides which tools are allowed for a finance assistant.

        Rules:
        - Only choose tools from TOOL LIST.
        - Choose only tools needed to answer the user's request.
        - If no tools are needed, return an empty list.
        - If the user asks about their portfolio or holdings, allow portfolio tools.
        - If the user asks for company analysis with explicit tickers, allow fundamentals/SEC tools.
        - If the user asks for market or company news, allow the news tool.
        - If personalization would help, allow the user profile tool.

        TOOL LIST:
        {tool_manifest}

        CHAT HISTORY:
        {history_block}

        USER MESSAGE:
        {message}

        INTENT:
        {intent}

        SYMBOLS:
        {symbols}

        """

    try:
        res = await structured_llm.ainvoke(prompt)
        allowed_tools = [t for t in res.allowed_tools if t in TOOL_REGISTRY]
    except Exception as exc:
        logger.warning("planner failed: %s", exc)
        allowed_tools = []
        if needs_user_profile:
            allowed_tools.append("get_user_profile")
        if needs_portfolio or intent == "portfolio":
            allowed_tools.append("get_portfolio_summary")
            allowed_tools.append("get_holdings")
        if symbols:
            allowed_tools.append("get_fundamentals")
            allowed_tools.append("get_sec_snippets")
        if intent == "market_news":
            allowed_tools.append("get_news")

    if not symbols:
        allowed_tools = [t for t in allowed_tools if t not in {"get_fundamentals", "get_sec_snippets"}]

    tool_caps = _default_tool_caps(symbols)
    debug = {
        "symbols": symbols,
        "intent": intent,
        "allowed_tools": allowed_tools,
        "tool_caps": tool_caps,
    }

    merged_debug = dict(state.get("debug") or {})
    merged_debug.update(debug)
    return {
        "allowed_tools": allowed_tools,
        "tool_caps": tool_caps,
        "debug": merged_debug,
    }


def _filter_tool_calls(
    calls: List[Any],
    allowed_tools: List[str],
    tool_caps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not calls:
        return []
    max_calls: Dict[str, int] = {}
    for name in allowed_tools:
        caps = tool_caps.get(name) or {}
        max_calls[name] = int(caps.get("max_calls", 1))
    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for call in calls:
        name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
        if not name or name not in allowed_tools:
            continue
        if counts.get(name, 0) >= max_calls.get(name, 1):
            continue
        args = getattr(call, "arguments", None)
        if args is None and isinstance(call, dict):
            args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        out.append({"name": name, "arguments": args})
        counts[name] = counts.get(name, 0) + 1
    return out


async def tool_select_node(state: ChatState) -> Dict[str, Any]:
    allowed_tools = state.get("allowed_tools") or []
    if not allowed_tools:
        merged_debug = dict(state.get("debug") or {})
        merged_debug.update({"tool_calls": 0})
        return {"tool_calls": [], "debug": merged_debug}

    message = (state.get("message") or "").strip()
    intent = (state.get("intent") or "").strip()
    symbols = state.get("symbols") or []
    history_block = _format_history(state.get("history") or [])
    tool_caps = state.get("tool_caps") or {}
    tool_manifest = render_tool_manifest(allowed_tools)

    structured_llm = llm.with_structured_output(ToolSelection, method="function_calling")
    prompt = f"""
        You are selecting tools for a finance assistant.

        Rules:
        - Use only tools in ALLOWED TOOLS.
        - Call tools only if needed to answer.
        - Respect tool caps; do not exceed max_calls.
        - Return a JSON object with: {{ "calls": [{{"name": "...", "arguments": {{...}}}}] }}.
        - If no tools are needed, return {{ "calls": [] }}.

        ALLOWED TOOLS:
        {tool_manifest}

        TOOL CAPS:
        {json.dumps(tool_caps, separators=(",", ":"))}

        CHAT HISTORY:
        {history_block}

        USER MESSAGE:
        {message}

        INTENT:
        {intent}

        SYMBOLS:
        {symbols}
        """

    res = await structured_llm.ainvoke(prompt)
    calls = _filter_tool_calls(res.calls, allowed_tools, tool_caps)
    merged_debug = dict(state.get("debug") or {})
    merged_debug.update({"tool_calls": len(calls)})
    return {
        "tool_calls": calls,
        "debug": merged_debug,
    }


async def tool_exec_node(state: ChatState) -> Dict[str, Any]:
    allowed_tools = state.get("allowed_tools") or []
    tool_calls = state.get("tool_calls") or []
    tool_caps = state.get("tool_caps") or {}

    ctx = ToolContext(
        db=state.get("db"),
        finnhub=state.get("finnhub"),
        user_id=state.get("user_id"),
        user_currency=state.get("user_currency") or "USD",
        message=state.get("message") or "",
        symbols=state.get("symbols") or [],
    )

    tasks: List[asyncio.Task[Any]] = []
    meta: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for call in tool_calls:
        name = (call.get("name") or "").strip()
        if not name or name not in allowed_tools:
            continue
        spec = TOOL_REGISTRY.get(name)
        if not spec:
            errors.append({"name": name, "error": "unknown_tool"})
            continue
        raw_args = call.get("arguments") or {}
        caps = tool_caps.get(name) or {}
        prepared_args = prepare_tool_args(name, raw_args, caps, state)
        validated_args = validate_tool_args(name, prepared_args)
        if not validated_args:
            errors.append({"name": name, "error": "invalid_args", "arguments": prepared_args})
            continue
        tasks.append(asyncio.create_task(spec.run(validated_args, ctx)))
        meta.append({"name": name, "arguments": prepared_args})

    results: List[Dict[str, Any]] = []
    if tasks:
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        for output, info in zip(outputs, meta):
            name = info.get("name")
            args = info.get("arguments") or {}
            if isinstance(output, Exception):
                errors.append({"name": name, "error": str(output), "arguments": args})
                continue
            results.append({"name": name, "arguments": args, "data": _json_safe(output)})

    debug = {
        "tool_results": len(results),
        "tool_errors": len(errors),
    }
    merged_debug = dict(state.get("debug") or {})
    merged_debug.update(debug)
    return {
        "tool_results": results,
        "tool_errors": errors,
        "debug": merged_debug,
    }


def _aggregate_tool_results(tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    portfolio_summary: Dict[str, Any] = {}
    holdings: List[Dict[str, Any]] = []
    fundamentals: Dict[str, Any] = {}
    fundamentals_gaps: Dict[str, List[str]] = {}
    sec_snippets: Dict[str, List[Dict[str, Any]]] = {"general": [], "business": [], "risk": [], "mda": []}
    news_items: List[Dict[str, Any]] = []

    for item in tool_results or []:
        name = item.get("name")
        data = item.get("data") or {}
        if name == "get_user_profile":
            if isinstance(data, dict):
                profile = data
        elif name == "get_portfolio_summary":
            if isinstance(data, dict):
                portfolio_summary = data
        elif name == "get_holdings":
            if isinstance(data, list):
                holdings = data
        elif name == "get_fundamentals":
            if isinstance(data, dict):
                fundamentals.update(data.get("fundamentals") or {})
                fundamentals_gaps.update(data.get("gaps") or {})
        elif name == "get_sec_snippets":
            if isinstance(data, dict):
                section = (data.get("section") or "general").strip().lower()
                snippets = data.get("snippets") or []
                if isinstance(snippets, list) and section in sec_snippets:
                    sec_snippets[section].extend(snippets)
        elif name == "get_news":
            if isinstance(data, dict):
                query = data.get("query") or ""
                items = data.get("items") or []
                if isinstance(items, list):
                    for entry in items:
                        if not isinstance(entry, dict):
                            continue
                        payload = dict(entry)
                        if query:
                            payload["query"] = query
                        news_items.append(payload)

    return {
        "user_profile": profile,
        "portfolio_summary": portfolio_summary,
        "holdings": holdings,
        "fundamentals": fundamentals,
        "fundamentals_gaps": fundamentals_gaps,
        "sec_snippets": sec_snippets,
        "news_items": news_items,
    }


async def answer_node(state: ChatState) -> Dict[str, Any]:
    tool_results = state.get("tool_results") or []

    intent = state.get("intent") or ""
    message = state.get("message") or ""

    if intent == "off_topic":
        return {
            "answer": "I can only help with stocks, crypto, portfolio questions, and market analysis. "
                      "If you have a finance question, ask away.",
        }

    tool_data = _aggregate_tool_results(tool_results)
    holdings = tool_data.get("holdings") or []
    if intent == "portfolio" and holdings and _HOLDINGS_Q_RE.search(message):
        return {
            "answer": _render_holdings_answer(holdings),
        }

    messages = _build_answer_messages(state, tool_results)
    res = await llm.ainvoke(messages)
    return {
        "answer": (res.content or "").strip(),
    }


def _route_next(state: ChatState) -> str:
    if (state.get("intent") or "").strip().lower() == "off_topic":
        return "off_topic"
    return "plan"


def _fast_next(state: ChatState) -> str:
    if state.get("short_circuit"):
        return "end"
    return "route"


workflow = StateGraph(ChatState)
workflow.add_node("fast_path", fast_path_node)
workflow.add_node("route", route_node)
workflow.add_node("plan", plan_node)
workflow.add_node("tool_select", tool_select_node)
workflow.add_node("tool_exec", tool_exec_node)
workflow.add_node("answer", answer_node)
workflow.add_node("off_topic", answer_node)

workflow.set_entry_point("fast_path")
workflow.add_conditional_edges("fast_path", _fast_next, {"route": "route", "end": END})
workflow.add_conditional_edges("route", _route_next, {"plan": "plan", "off_topic": "off_topic"})
workflow.add_edge("plan", "tool_select")
workflow.add_edge("tool_select", "tool_exec")
workflow.add_edge("tool_exec", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("off_topic", END)

app_graph = workflow.compile()


async def run_chat_turn(
    *,
    message: str,
    user_id: Any,
    user_currency: str,
    session_id: str,
    history: List[Dict[str, str]],
    db: Session,
    finnhub: Any,
) -> Tuple[str, Dict[str, Any]]:
    initial_state: ChatState = {
        "message": message,
        "user_id": user_id,
        "user_currency": user_currency,
        "session_id": session_id,
        "history": history,
        "db": db,
        "finnhub": finnhub,
    }

    final_state = await app_graph.ainvoke(initial_state)
    answer = final_state.get("answer") or ""
    debug = final_state.get("debug") or {}
    return answer, debug


async def run_chat_turn_stream(
    *,
    message: str,
    user_id: Any,
    user_currency: str,
    session_id: str,
    history: List[Dict[str, str]],
    db: Session,
    finnhub: Any,
):
    state: ChatState = {
        "message": message,
        "user_id": user_id,
        "user_currency": user_currency,
        "session_id": session_id,
        "history": history,
        "db": db,
        "finnhub": finnhub,
    }

    state.update(await fast_path_node(state))
    if state.get("short_circuit"):
        answer = state.get("answer") or ""
        if answer:
            yield answer
        return

    state.update(await route_node(state))
    if (state.get("intent") or "").strip().lower() == "off_topic":
        answer = (
            "I can only help with stocks, crypto, portfolio questions, and market analysis. "
            "If you have a finance question, ask away."
        )
        if answer:
            yield answer
        return

    state.update(await plan_node(state))
    state.update(await tool_select_node(state))
    state.update(await tool_exec_node(state))

    tool_data = _aggregate_tool_results(state.get("tool_results") or [])
    holdings = tool_data.get("holdings") or []
    if (state.get("intent") or "") == "portfolio" and holdings and _HOLDINGS_Q_RE.search(message):
        answer = _render_holdings_answer(holdings)
        if answer:
            yield answer
        return

    messages = _build_answer_messages(state, state.get("tool_results") or [])
    answer_parts: List[str] = []
    async for chunk in llm_stream.astream(messages):
        text = (chunk.content or "")
        if not text:
            continue
        answer_parts.append(text)
        yield text
