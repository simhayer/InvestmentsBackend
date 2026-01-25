from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.cache.cache_backend import cache_get, cache_set
from services.holding_service import get_holdings_with_live_prices
from services.ai.chat_agent.tools import (
    TOOL_REGISTRY,
    ToolContext,
    ToolSelection,
    prepare_tool_args,
    render_tool_manifest,
    validate_tool_args,
)
from .types import ChatState

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
    tool_errors: List[Dict[str, Any]] | None = None,
) -> List[Any]:
    intent = (state.get("intent") or "").strip()
    message = state.get("message") or ""
    symbols = state.get("symbols") or []

    history_block = ""
    for item in tool_results or []:
        if item.get("name") != "get_chat_history":
            continue
        data = item.get("data") or {}
        if isinstance(data, dict):
            text = (data.get("text") or "").strip()
            if text:
                history_block = text
                break
            messages = data.get("messages")
            if isinstance(messages, list):
                history_block = _format_history(messages)
                break
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

    error_lines: List[str] = []
    for err in tool_errors or []:
        if not isinstance(err, dict):
            continue
        name = (err.get("name") or "").strip()
        detail = (err.get("error") or "").strip()
        if not name:
            continue
        error_lines.append(f"{name}: {detail}" if detail else name)
    errors_block = "\n".join(error_lines) if error_lines else "None"

    history_section = f"CHAT HISTORY:\n{history_block}\n\n" if history_block else ""
    intent_section = f"INTENT:\n{intent}\n\n" if intent else ""
    symbols_section = f"SYMBOLS:\n{symbols}\n\n" if symbols else ""

    user_content = f"""{history_section}USER MESSAGE:
{message}

{intent_section}{symbols_section}TOOL ERRORS:
{errors_block}
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


def _default_tool_caps() -> Dict[str, Any]:
    return {
        "get_user_profile": {"max_calls": 1},
        "get_portfolio_summary": {"max_calls": 1, "top_n": 5},
        "get_holdings": {"max_calls": 1, "max_items": MAX_HOLDINGS},
        "get_fundamentals": {"max_calls": 1, "max_symbols": MAX_SYMBOLS},
        "get_sec_snippets": {"max_calls": 2, "max_snippets": 6},
        "get_news": {"max_calls": 1, "max_results": 6},
        "get_chat_history": {"max_calls": 1, "max_items": CHAT_MAX_HISTORY},
    }


async def plan_node(state: ChatState) -> Dict[str, Any]:
    message = (state.get("message") or "").strip()

    tool_manifest = render_tool_manifest(list(TOOL_REGISTRY.keys()))

    structured_llm = planner_llm.with_structured_output(ToolPlan, method="function_calling")
    prompt = f"""
        You are a planner that decides which tools are allowed for a finance assistant.

        Rules:
        - Only choose tools from TOOL LIST.
        - Choose only tools needed to answer the user's request.
        - If no tools are needed, return an empty list.
        - Avoid portfolio tools unless the user asks about their own holdings or portfolio.
        - Allow fundamentals/SEC tools only for company or ticker-specific requests.
        - Allow the news tool for market or company news requests.
        - Allow the user profile tool when personalization would help.
        - Allow the chat history tool only when the user asks to recap or references earlier context.

        TOOL LIST:
        {tool_manifest}

        USER MESSAGE:
        {message}

        """

    try:
        res = await structured_llm.ainvoke(prompt)
        allowed_tools = [t for t in res.allowed_tools if t in TOOL_REGISTRY]
    except Exception as exc:
        logger.warning("planner failed: %s", exc)
        allowed_tools = []

    tool_caps = _default_tool_caps()
    debug = {
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
        - Use get_chat_history only when the user asks for recap or refers to earlier context.
        - When calling get_fundamentals, include "symbols": ["TICKER", ...].
        - When calling get_sec_snippets, include "symbol" and "section".
        - When calling get_news, include a short "query" if the user asks for news.

        ALLOWED TOOLS:
        {tool_manifest}

        TOOL CAPS:
        {json.dumps(tool_caps, separators=(",", ":"))}

        USER MESSAGE:
        {message}
        """

    try:
        res = await structured_llm.ainvoke(prompt)
        calls = _filter_tool_calls(res.calls, allowed_tools, tool_caps)
    except Exception as exc:
        logger.warning("tool_select failed: %s", exc)
        calls = []
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

    tasks: List[asyncio.Task[Any]] = []
    meta: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    prepared_calls: List[Dict[str, Any]] = []

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
        prepared_calls.append(
            {
                "name": name,
                "arguments": prepared_args,
                "validated": validated_args,
                "spec": spec,
            }
        )

    holdings_snapshot = None
    if any(c.get("name") in {"get_holdings", "get_portfolio_summary"} for c in prepared_calls):
        if state.get("db") is not None and state.get("finnhub") is not None and state.get("user_id") is not None:
            top_n = 0
            for call in prepared_calls:
                if call.get("name") != "get_portfolio_summary":
                    continue
                try:
                    top_n = max(top_n, int(call.get("arguments", {}).get("top_n", 0) or 0))
                except (TypeError, ValueError):
                    continue
            try:
                holdings_snapshot = await get_holdings_with_live_prices(
                    user_id=str(state.get("user_id")),
                    db=state.get("db"),
                    finnhub=state.get("finnhub"),
                    currency=state.get("user_currency") or "USD",
                    top_only=False,
                    top_n=top_n,
                    include_weights=True,
                )
            except Exception as exc:
                logger.warning("holdings snapshot failed: %s", exc)

    ctx = ToolContext(
        db=state.get("db"),
        finnhub=state.get("finnhub"),
        user_id=state.get("user_id"),
        user_currency=state.get("user_currency") or "USD",
        message=state.get("message") or "",
        symbols=state.get("symbols") or [],
        history=state.get("history") or [],
        holdings_snapshot=holdings_snapshot,
    )

    for call in prepared_calls:
        spec = call.get("spec")
        if not spec:
            continue
        tasks.append(asyncio.create_task(spec.run(call.get("validated"), ctx)))
        meta.append({"name": call.get("name"), "arguments": call.get("arguments")})

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


async def answer_node(state: ChatState) -> Dict[str, Any]:
    tool_results = state.get("tool_results") or []
    tool_errors = state.get("tool_errors") or []

    messages = _build_answer_messages(state, tool_results, tool_errors)
    res = await llm.ainvoke(messages)
    return {
        "answer": (res.content or "").strip(),
    }


workflow = StateGraph(ChatState)
workflow.add_node("plan", plan_node)
workflow.add_node("tool_select", tool_select_node)
workflow.add_node("tool_exec", tool_exec_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "tool_select")
workflow.add_edge("tool_select", "tool_exec")
workflow.add_edge("tool_exec", "answer")
workflow.add_edge("answer", END)

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

    state.update(await plan_node(state))
    state.update(await tool_select_node(state))
    state.update(await tool_exec_node(state))

    messages = _build_answer_messages(
        state,
        state.get("tool_results") or [],
        state.get("tool_errors") or [],
    )
    answer_parts: List[str] = []
    async for chunk in llm_stream.astream(messages):
        text = (chunk.content or "")
        if not text:
            continue
        answer_parts.append(text)
        yield text
