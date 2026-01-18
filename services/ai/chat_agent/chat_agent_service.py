from __future__ import annotations

import asyncio
import hashlib
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
from services.holding_service import get_all_holdings, get_holdings_with_live_prices
from services.portfolio_service import get_portfolio_summary
from services.tavily.client import search as tavily_search, compact_results as compact_tavily
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
MAX_HOLDINGS = int(os.getenv("CHAT_MAX_HOLDINGS", "25"))
TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))

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
_NEWS_HINTS = {
    "news",
    "headline",
    "headlines",
    "catalyst",
    "catalysts",
    "event",
    "events",
    "earnings",
    "guidance",
    "downgrade",
    "upgrade",
    "lawsuit",
    "investigation",
}
_FUNDAMENTAL_HINTS = {
    "fundamentals",
    "financials",
    "valuation",
    "metrics",
    "margin",
    "revenue",
    "earnings",
    "profit",
    "balance sheet",
    "cash flow",
}
_SEC_BUSINESS_HINTS = {
    "business",
    "model",
    "segments",
    "customers",
    "products",
    "revenue",
    "moat",
}
_SEC_GENERAL_HINTS = {
    "sec",
    "filing",
    "filings",
    "10-k",
    "10k",
    "10-q",
    "10q",
    "8-k",
    "8k",
}
_SEC_RISK_HINTS = {
    "risk",
    "risks",
    "uncertainty",
    "headwinds",
    "competition",
    "regulation",
    "liquidity",
    "debt",
    "margin",
}
_SEC_MDA_HINTS = {
    "md&a",
    "mda",
    "outlook",
    "guidance",
    "trend",
    "trends",
    "management discussion",
}
_HOLDINGS_DETAIL_HINTS = {
    "holdings",
    "positions",
    "allocation",
    "weights",
    "diversif",
    "exposure",
    "sector",
    "what do i own",
    "what do we own",
    "list",
    "show",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


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


def _ck_news(q: str) -> str:
    digest = hashlib.sha256((q or "").strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"CHAT:NEWS:{digest}"


def _holding_brief(item: Any) -> Dict[str, Any]:
    data = _json_safe(item)
    if not isinstance(data, dict):
        return {}
    keys = [
        "symbol",
        "name",
        "type",
        "quantity",
        "current_price",
        "value",
        "currency",
        "weight",
        "unrealized_pl",
        "unrealized_pl_pct",
        "account_name",
    ]
    return {k: data.get(k) for k in keys}


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


async def context_plan_node(state: ChatState) -> Dict[str, Any]:
    message = (state.get("message") or "").strip()
    intent = (state.get("intent") or "").strip()
    symbols = state.get("symbols") or []
    needs_portfolio = bool(state.get("needs_portfolio"))
    needs_user_profile = bool(state.get("needs_user_profile"))

    msg_lower = message.lower()
    mentions_portfolio = "portfolio" in msg_lower or "holdings" in msg_lower or "positions" in msg_lower
    holdings_detail = bool(
        _HOLDINGS_Q_RE.search(message)
        or _LIST_RE.search(message)
        or _contains_any(message, _HOLDINGS_DETAIL_HINTS)
    )

    fetch_portfolio_summary = needs_portfolio or intent == "portfolio" or mentions_portfolio
    fetch_holdings = fetch_portfolio_summary and holdings_detail
    fetch_user_profile = needs_user_profile

    fetch_fundamentals = bool(symbols) and (
        intent in {"stock_analysis", "education", "portfolio"}
        or _contains_any(message, _FUNDAMENTAL_HINTS)
    )
    fetch_sec_context = bool(symbols) and (
        intent in {"stock_analysis", "education", "portfolio"}
        or _contains_any(message, _SEC_GENERAL_HINTS)
    )
    fetch_sec_business = fetch_sec_context and _contains_any(message, _SEC_BUSINESS_HINTS)
    fetch_sec_risk = fetch_sec_context and _contains_any(message, _SEC_RISK_HINTS)
    fetch_sec_mda = fetch_sec_context and _contains_any(message, _SEC_MDA_HINTS)

    fetch_news = False
    if intent == "market_news":
        fetch_news = True
    elif _contains_any(message, _NEWS_HINTS):
        fetch_news = True

    return {
        "fetch_user_profile": fetch_user_profile,
        "fetch_portfolio_summary": fetch_portfolio_summary,
        "fetch_holdings": fetch_holdings,
        "fetch_fundamentals": fetch_fundamentals,
        "fetch_sec_context": fetch_sec_context,
        "fetch_sec_business": fetch_sec_business,
        "fetch_sec_risk": fetch_sec_risk,
        "fetch_sec_mda": fetch_sec_mda,
        "fetch_news": fetch_news,
    }


async def research_node(state: ChatState) -> Dict[str, Any]:
    db: Session = state.get("db")
    finnhub = state.get("finnhub")
    user_id = state.get("user_id")
    message = state.get("message") or ""
    symbols = state.get("symbols") or []
    intent = state.get("intent") or ""

    fetch_user_profile = bool(state.get("fetch_user_profile") or state.get("needs_user_profile"))
    fetch_portfolio_summary = bool(state.get("fetch_portfolio_summary") or state.get("needs_portfolio"))
    fetch_holdings = bool(state.get("fetch_holdings"))
    fetch_fundamentals = bool(state.get("fetch_fundamentals"))
    fetch_sec_context = bool(state.get("fetch_sec_context"))
    fetch_sec_business = bool(state.get("fetch_sec_business"))
    fetch_sec_risk = bool(state.get("fetch_sec_risk"))
    fetch_sec_mda = bool(state.get("fetch_sec_mda"))
    fetch_news = bool(state.get("fetch_news"))

    user_profile: Dict[str, Any] = {}
    if fetch_user_profile and db is not None and user_id is not None:
        profile = (
            db.query(UserOnboardingProfile)
            .filter(UserOnboardingProfile.user_id == user_id)
            .first()
        )
        user_profile = _serialize_onboarding(profile)

    portfolio_summary: Dict[str, Any] = {}
    holdings: List[Dict[str, Any]] = []
    if (fetch_portfolio_summary or fetch_holdings) and finnhub is not None and db is not None:
        try:
            currency = (state.get("user_currency") or "USD").upper()
            if fetch_portfolio_summary:
                portfolio_summary = await get_portfolio_summary(
                    user_id=str(user_id),
                    db=db,
                    finnhub=finnhub,
                    currency=currency,
                    top_n=5,
                )
            if fetch_holdings:
                holdings_payload = await get_holdings_with_live_prices(
                    user_id=str(user_id),
                    db=db,
                    finnhub=finnhub,
                    currency=currency,
                    top_only=False,
                    top_n=0,
                    include_weights=True,
                )
                items = (holdings_payload or {}).get("items") or []
                holdings = [_holding_brief(it) for it in items if it]  # type: ignore[arg-type]
                holdings = [h for h in holdings if h.get("symbol")][:MAX_HOLDINGS]
        except Exception as exc:
            logger.warning("portfolio_summary failed: %s", exc)

    fundamentals: Dict[str, Any] = {}
    fundamentals_gaps: Dict[str, List[str]] = {}
    if fetch_fundamentals and symbols:
        tasks = [fetch_fundamentals_cached(sym, timeout_s=5.0) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                fundamentals_gaps[sym] = [str(res)]
                continue
            fundamentals[sym] = res.data
            fundamentals_gaps[sym] = res.gaps

    vector_context = ""
    sec_business_context = ""
    sec_risk_context = ""
    sec_mda_context = ""
    if symbols and intent != "crypto" and (fetch_sec_context or fetch_sec_business or fetch_sec_risk or fetch_sec_mda):
        vector_service = VectorStoreService()
        chunks: List[str] = []
        business_chunks: List[str] = []
        risk_chunks: List[str] = []
        mda_chunks: List[str] = []
        for sym in symbols:
            try:
                if fetch_sec_business:
                    business_sec = vector_service.get_context_for_analysis(
                        db=db,
                        symbol=sym,
                        query="business model, segments, products, customers, revenue drivers",
                        limit=6,
                    )
                    if business_sec:
                        body = "\n".join(
                            f"- ({c.get('metadata', {}).get('form_type','?')} {c.get('metadata', {}).get('filed_date','?')}) {c.get('content')}"
                            for c in business_sec
                        )
                        business_chunks.append(f"{sym} SEC BUSINESS CONTEXT:\n{body}")

                if fetch_sec_risk:
                    risk_sec = vector_service.get_context_for_analysis(
                        db=db,
                        symbol=sym,
                        query="risk factors, competition, regulation, liquidity, debt, margin pressure",
                        limit=6,
                    )
                    if risk_sec:
                        body = "\n".join(
                            f"- ({c.get('metadata', {}).get('form_type','?')} {c.get('metadata', {}).get('filed_date','?')}) {c.get('content')}"
                            for c in risk_sec
                        )
                        risk_chunks.append(f"{sym} SEC RISK CONTEXT:\n{body}")

                if fetch_sec_mda:
                    mda_sec = vector_service.get_context_for_analysis(
                        db=db,
                        symbol=sym,
                        query="management discussion and analysis, outlook, trends, guidance, MD&A",
                        limit=6,
                    )
                    if mda_sec:
                        body = "\n".join(
                            f"- ({c.get('metadata', {}).get('form_type','?')} {c.get('metadata', {}).get('filed_date','?')}) {c.get('content')}"
                            for c in mda_sec
                        )
                        mda_chunks.append(f"{sym} SEC MD&A CONTEXT:\n{body}")

                if fetch_sec_context:
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
        sec_business_context = "\n\n".join(business_chunks)[:MAX_VECTOR_CHARS]
        sec_risk_context = "\n\n".join(risk_chunks)[:MAX_VECTOR_CHARS]
        sec_mda_context = "\n\n".join(mda_chunks)[:MAX_VECTOR_CHARS]

    news_context = ""
    if fetch_news:
        query = ""
        if symbols:
            query = f"latest news, catalysts, and risks for {symbols[0]}"
        elif intent == "market_news":
            query = f"latest market news and catalysts: {message}"
        else:
            query = f"latest news related to: {message}"
        if query:
            news_key = _ck_news(query)
            cached = cache_get(news_key)
            if isinstance(cached, str):
                news_context = cached
            else:
                try:
                    results = await tavily_search(
                        query=query,
                        max_results=6,
                        include_answer=False,
                        include_raw_content=False,
                        search_depth="advanced",
                    )
                    news_context = compact_tavily(results)
                    cache_set(news_key, news_context, ttl_seconds=TTL_TAVILY_SEC)
                except Exception as exc:
                    logger.warning("tavily search failed: %s", exc)

    debug = {
        "symbols": symbols,
        "intent": intent,
        "fetch_user_profile": fetch_user_profile,
        "fetch_portfolio_summary": fetch_portfolio_summary,
        "fetch_holdings": fetch_holdings,
        "fetch_fundamentals": fetch_fundamentals,
        "fetch_sec_context": fetch_sec_context,
        "fetch_sec_business": fetch_sec_business,
        "fetch_sec_risk": fetch_sec_risk,
        "fetch_sec_mda": fetch_sec_mda,
        "fetch_news": fetch_news,
        "has_portfolio": bool(portfolio_summary),
        "holdings_count": len(holdings),
        "has_profile": bool(user_profile),
        "vector_chars": len(vector_context),
        "sec_business_chars": len(sec_business_context),
        "sec_risk_chars": len(sec_risk_context),
        "sec_mda_chars": len(sec_mda_context),
        "news_chars": len(news_context),
        "fundamentals_symbols": list(fundamentals.keys()),
    }

    return {
        "user_profile": user_profile,
        "portfolio_summary": _json_safe(portfolio_summary),
        "holdings": holdings,
        "fundamentals": fundamentals,
        "fundamentals_gaps": fundamentals_gaps,
        "vector_context": vector_context,
        "sec_business_context": sec_business_context,
        "sec_risk_context": sec_risk_context,
        "sec_mda_context": sec_mda_context,
        "news_context": news_context,
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

    holdings = state.get("holdings") or []
    if intent == "portfolio" and holdings and _HOLDINGS_Q_RE.search(message):
        return {
            "answer": _render_holdings_answer(holdings),
            "iterations": int(state.get("iterations", 0)) + 1,
        }

    critique = (state.get("critique") or "").strip()
    critique_block = f"\n\nCRITIQUE TO FIX:\n{critique}\n" if critique else ""

    history_block = _format_history(history)
    profile_block = json.dumps(_json_safe(state.get("user_profile") or {}), separators=(",", ":"))
    portfolio_block = json.dumps(_json_safe(state.get("portfolio_summary") or {}), separators=(",", ":"))
    holdings_block = json.dumps(_json_safe(state.get("holdings") or []), separators=(",", ":"))
    fundamentals_block = json.dumps(_json_safe(state.get("fundamentals") or {}), separators=(",", ":"))
    gaps_block = json.dumps(_json_safe(state.get("fundamentals_gaps") or {}), separators=(",", ":"))
    vector_context = state.get("vector_context") or ""
    sec_business_context = state.get("sec_business_context") or ""
    sec_risk_context = state.get("sec_risk_context") or ""
    sec_mda_context = state.get("sec_mda_context") or ""
    news_context = state.get("news_context") or ""
    prompt = f"""
        You are a senior financial analyst and stock expert.
        Stay strictly within stocks, ETFs, crypto, portfolio, and market analysis.

        Guardrails:
        - Do NOT invent numbers. If data is missing, say "Not available".
        - Use SEC vector context as authoritative for company fundamentals and risks.
        - Use fundamentals data (if present) for concrete metrics.
        - If the user asks about a company but no explicit ticker is provided, ask for the ticker.
        - Avoid trading instructions. Provide analysis and education only.
        - If portfolio data is provided, only reference those holdings.

        Personalization:
        - Tailor tone to the user's onboarding profile when present (risk level, time horizon, experience).
        - Include only sections that have supporting data (skip empty sections).
        - If news_context is provided, include a short "News" section grounded in those items.
        - If SEC context is provided, include a short "SEC" section (business/risks/MD&A as available).
        - If portfolio data is provided, include a "Portfolio" section with concrete holdings/summary.
        - If fundamentals are provided, include key metrics and note gaps when relevant.

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

        HOLDINGS:
        {holdings_block}

        FUNDAMENTALS:
        {fundamentals_block}

        FUNDAMENTALS GAPS:
        {gaps_block}

        VECTOR CONTEXT (SEC filings):
        {vector_context}

        SEC BUSINESS SNIPS:
        {sec_business_context}

        SEC RISK SNIPS:
        {sec_risk_context}

        SEC MD&A SNIPS:
        {sec_mda_context}

        NEWS CONTEXT:
        {news_context}
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
    return "plan"


def _fast_next(state: ChatState) -> str:
    if state.get("short_circuit"):
        return "end"
    return "route"


workflow = StateGraph(ChatState)
workflow.add_node("fast_path", fast_path_node)
workflow.add_node("route", route_node)
workflow.add_node("plan", context_plan_node)
workflow.add_node("research", research_node)
workflow.add_node("answer", answer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("off_topic", answer_node)

workflow.set_entry_point("fast_path")
workflow.add_conditional_edges("fast_path", _fast_next, {"route": "route", "end": END})
workflow.add_conditional_edges("route", _route_next, {"plan": "plan", "off_topic": "off_topic"})
workflow.add_edge("plan", "research")
workflow.add_edge("research", "answer")
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
