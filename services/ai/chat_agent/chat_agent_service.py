from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.portfolio_service import get_portfolio_summary
from services.vector.vector_store_service import VectorStoreService
from models.user_onboarding_profile import UserOnboardingProfile
from .types import ChatState, IntentResult

logger = logging.getLogger("chat_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.2)

CHAT_HISTORY_TTL_SEC = int(os.getenv("CHAT_HISTORY_TTL_SEC", "21600"))  # 6h
CHAT_MAX_HISTORY = int(os.getenv("CHAT_MAX_HISTORY", "14"))
MAX_SYMBOLS = int(os.getenv("CHAT_MAX_SYMBOLS", "3"))
MAX_VECTOR_CHARS = int(os.getenv("CHAT_MAX_VECTOR_CHARS", "6000"))

_EXPLICIT_SYMBOL_RE = re.compile(r"\$([A-Za-z]{1,6})\b")


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


def _serialize_onboarding(profile: UserOnboardingProfile | None) -> Dict[str, Any]:
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


async def context_node(state: ChatState) -> Dict[str, Any]:
    db: Session = state.get("db")
    finnhub = state.get("finnhub")
    user_id = state.get("user_id")
    message = state.get("message") or ""
    symbols = state.get("symbols") or []
    intent = state.get("intent") or ""

    user_profile: Dict[str, Any] = {}
    if state.get("needs_user_profile") and db is not None and user_id is not None:
        profile = (
            db.query(UserOnboardingProfile)
            .filter(UserOnboardingProfile.user_id == user_id)
            .first()
        )
        user_profile = _serialize_onboarding(profile)

    portfolio_summary: Dict[str, Any] = {}
    if state.get("needs_portfolio") and finnhub is not None and db is not None:
        try:
            currency = (state.get("user_currency") or "USD").upper()
            portfolio_summary = await get_portfolio_summary(
                user_id=str(user_id),
                db=db,
                finnhub=finnhub,
                currency=currency,
                top_n=5,
            )
        except Exception as exc:
            logger.warning("portfolio_summary failed: %s", exc)

    fundamentals: Dict[str, Any] = {}
    fundamentals_gaps: Dict[str, List[str]] = {}
    if symbols:
        tasks = [fetch_fundamentals_cached(sym, timeout_s=5.0) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                fundamentals_gaps[sym] = [str(res)]
                continue
            fundamentals[sym] = res.data
            fundamentals_gaps[sym] = res.gaps

    vector_context = ""
    if symbols and intent != "crypto":
        vector_service = VectorStoreService()
        chunks: List[str] = []
        for sym in symbols:
            try:
                sec_chunks = vector_service.get_context_for_analysis(
                    db=db,
                    symbol=sym,
                    query=message,
                    limit=6,
                )
                if sec_chunks:
                    body = "\n".join(
                        f"- ({c.get('metadata', {}).get('form_type','?')} {c.get('metadata', {}).get('filed_date','?')}) {c.get('content')}"
                        for c in sec_chunks
                    )
                    chunks.append(f"{sym} SEC CONTEXT:\n{body}")
            except Exception as exc:
                logger.warning("vector_context failed for %s: %s", sym, exc)
        vector_context = "\n\n".join(chunks)[:MAX_VECTOR_CHARS]

    debug = {
        "symbols": symbols,
        "intent": intent,
        "has_portfolio": bool(portfolio_summary),
        "has_profile": bool(user_profile),
        "vector_chars": len(vector_context),
        "fundamentals_symbols": list(fundamentals.keys()),
    }

    return {
        "user_profile": user_profile,
        "portfolio_summary": portfolio_summary,
        "fundamentals": fundamentals,
        "fundamentals_gaps": fundamentals_gaps,
        "vector_context": vector_context,
        "debug": debug,
    }


async def answer_node(state: ChatState) -> Dict[str, Any]:
    intent = state.get("intent") or ""
    message = state.get("message") or ""
    history = state.get("history") or []
    symbols = state.get("symbols") or []

    if intent == "off_topic":
        return {
            "answer": "I can only help with stocks, crypto, portfolio questions, and market analysis. "
                      "If you have a finance question, ask away.",
            "iterations": int(state.get("iterations", 0)) + 1,
        }

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE TO FIX:\n{critique}\n" if critique else ""

    history_block = _format_history(history)
    profile_block = json.dumps(_json_safe(state.get("user_profile") or {}), separators=(",", ":"))
    portfolio_block = json.dumps(_json_safe(state.get("portfolio_summary") or {}), separators=(",", ":"))
    fundamentals_block = json.dumps(_json_safe(state.get("fundamentals") or {}), separators=(",", ":"))
    gaps_block = json.dumps(_json_safe(state.get("fundamentals_gaps") or {}), separators=(",", ":"))
    vector_context = state.get("vector_context") or ""

    prompt = f"""
        You are a senior financial analyst and stock expert.
        Stay strictly within stocks, ETFs, crypto, portfolio, and market analysis.

        Guardrails:
        - Do NOT invent numbers. If data is missing, say "Not available".
        - Use SEC vector context as authoritative for company fundamentals and risks.
        - Use fundamentals data (if present) for concrete metrics.
        - If the user asks about a company but no explicit ticker is provided, ask for the ticker.
        - Avoid trading instructions. Provide analysis and education only.

        Personalization:
        - Tailor tone to the user's onboarding profile when present (risk level, time horizon, experience).

        CHAT HISTORY:
        {history_block}

        USER MESSAGE:
        {message}

        SYMBOLS:
        {symbols}

        USER PROFILE:
        {profile_block}

        PORTFOLIO SUMMARY:
        {portfolio_block}

        FUNDAMENTALS:
        {fundamentals_block}

        FUNDAMENTALS GAPS:
        {gaps_block}

        VECTOR CONTEXT (SEC filings):
        {vector_context}
        {critique_block}

        Response style:
        - Be concise and structured.
        - Use short paragraphs or bullets.
        - End with a short prompt for follow-up if needed.
        """

    res = await llm.ainvoke(prompt)
    return {
        "answer": (res.content or "").strip(),
        "iterations": int(state.get("iterations", 0)) + 1,
    }


async def critic_node(state: ChatState) -> Dict[str, Any]:
    iters = int(state.get("iterations", 0))
    answer = state.get("answer") or ""

    if not answer:
        return {"is_valid": True, "critique": ""}

    prompt = f"""
        You are a strict reviewer for finance answers.
        Check for:
        - off-topic content
        - invented numbers or data not grounded in provided context
        - investment advice beyond analysis/education
        - contradictions with provided fundamentals or SEC context

        If it is good enough to ship, respond exactly: CLEAR
        Otherwise respond with 3-6 specific fixes.

        ANSWER:
        {answer}
        """

    res = await llm.ainvoke(prompt)
    txt = (res.content or "").strip()
    is_clear = txt.upper() == "CLEAR"

    if iters >= 1:
        return {"is_valid": True, "critique": "" if is_clear else txt}

    return {"is_valid": bool(is_clear), "critique": "" if is_clear else txt}


def _route_next(state: ChatState) -> str:
    if (state.get("intent") or "").strip().lower() == "off_topic":
        return "off_topic"
    return "context"


workflow = StateGraph(ChatState)
workflow.add_node("route", route_node)
workflow.add_node("context", context_node)
workflow.add_node("answer", answer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("off_topic", answer_node)

workflow.set_entry_point("route")
workflow.add_conditional_edges("route", _route_next, {"context": "context", "off_topic": "off_topic"})
workflow.add_edge("context", "answer")
workflow.add_edge("answer", "critic")
workflow.add_conditional_edges(
    "critic",
    lambda s: "end" if s.get("is_valid") else "revise",
    {"revise": "answer", "end": END},
)
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
        "iterations": 0,
    }

    final_state = await app_graph.ainvoke(initial_state)
    answer = final_state.get("answer") or ""
    debug = final_state.get("debug") or {}
    return answer, debug
