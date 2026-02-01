from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from models.user_onboarding_profile import UserOnboardingProfile
from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from services.holding_service import get_holdings_with_live_prices
from services.portfolio_service import get_portfolio_summary
from services.tavily.client import search as tavily_search
from services.vector.vector_store_service import VectorStoreService

logger = logging.getLogger("chat_agent.tools")

TTL_TAVILY_SEC = int(os.getenv("TTL_TAVILY_SEC", "900"))
TTL_WEB_SEARCH_SEC = int(os.getenv("TTL_WEB_SEARCH_SEC", str(TTL_TAVILY_SEC)))
TAVILY_WEB_SEARCH_DAYS = int(os.getenv("TAVILY_WEB_SEARCH_DAYS", "30"))
TAVILY_NEWS_DAYS = int(os.getenv("TAVILY_NEWS_DAYS", "7"))
TAVILY_WEB_DOMAINS = [
    d.strip()
    for d in (os.getenv("TAVILY_WEB_DOMAINS") or "").split(",")
    if d.strip()
]
MAX_NEWS_RESULTS = int(os.getenv("CHAT_MAX_NEWS_RESULTS", "6"))
MAX_SNIPPET_CHARS = int(os.getenv("CHAT_MAX_SEC_SNIPPET_CHARS", "360"))
MAX_QUERY_CHARS = int(os.getenv("CHAT_MAX_QUERY_CHARS", "240"))

ALLOWED_SEC_SECTIONS = {"general", "business", "risk", "mda"}


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolSelection(BaseModel):
    calls: List[ToolCall] = Field(default_factory=list)


class NoArgs(BaseModel):
    pass


class PortfolioSummaryInput(BaseModel):
    top_n: int = Field(default=5, ge=1, le=10)


class HoldingsInput(BaseModel):
    max_items: int = Field(default=25, ge=1, le=100)


class FundamentalsInput(BaseModel):
    symbols: List[str] = Field(default_factory=list)


class SecSnippetsInput(BaseModel):
    symbol: str
    section: Literal["general", "business", "risk", "mda"] = "general"
    query: str | None = None
    limit: int = Field(default=6, ge=1, le=10)


class NewsInput(BaseModel):
    symbols: Optional[List[str]] = None
    topic: Optional[str] = None  # e.g. "markets", "inflation", "bank of canada"
    query: Optional[str] = None  # override if you want exact
    max_results: int = Field(default=5, ge=1, le=10)
    days: int = Field(default=7, ge=1, le=30)

class PortfolioContextInput(BaseModel):
    top_n: int = Field(default=8, ge=1, le=15)
    max_holdings: int = Field(default=25, ge=5, le=100)

class PortfolioContextOutput(BaseModel):
    as_of: Optional[int] = None
    currency: str = "USD"
    portfolio_summary: Dict[str, Any] = Field(default_factory=dict)
    holdings: List[Dict[str, Any]] = Field(default_factory=list)
    symbols_for_news: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)

class WebSearchInput(BaseModel):
    query: str | None = None
    max_results: int = Field(default=5, ge=1, le=10)


class ChatHistoryInput(BaseModel):
    max_items: int = Field(default=10, ge=1, le=50)


@dataclass(frozen=True)
class ToolContext:
    db: Any
    finnhub: Any
    user_id: Any
    user_currency: str
    message: str
    symbols: List[str]
    history: List[Dict[str, str]] = field(default_factory=list)
    holdings_snapshot: Dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], Awaitable[Any]]


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


def _holding_brief(item: Any) -> Dict[str, Any]:
    data = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
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


def _normalize_symbols(symbols: Any) -> List[str]:
    out: List[str] = []
    if not symbols:
        return out
    for sym in symbols:
        if not sym or not isinstance(sym, str):
            continue
        cand = sym.strip().upper()
        if cand and cand not in out:
            out.append(cand)
    return out


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return default
    if num < min_value:
        return min_value
    if num > max_value:
        return max_value
    return num


def _clamp_text(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit]


def _news_cache_key(query: str) -> str:
    digest = hashlib.sha256((query or "").strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"CHAT:NEWS:{digest}"


def _web_search_cache_key(query: str) -> str:
    digest = hashlib.sha256((query or "").strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"CHAT:WEB:{digest}"


def _normalize_news_results(results: Any, limit: int) -> List[Dict[str, Any]]:
    items = []
    if isinstance(results, dict):
        raw_items = results.get("results") or results.get("data") or []
        if isinstance(raw_items, list):
            items = raw_items
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for item in items[: max(1, limit)]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or item.get("snippet") or "").strip()
        if content:
            content = content.replace("\n", " ").strip()
        payload = {
            "title": title,
            "url": url,
            "summary": _clamp_text(content, 280),
        }
        source = (item.get("source") or "").strip()
        if source:
            payload["source"] = source
        published = item.get("published_date") or item.get("published_at")
        if published:
            payload["published_at"] = published
        out.append(payload)
    return out


def _normalize_search_results(results: Any, limit: int) -> List[Dict[str, Any]]:
    items = []
    if isinstance(results, dict):
        raw_items = results.get("results") or results.get("data") or []
        if isinstance(raw_items, list):
            items = raw_items
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for item in items[: max(1, limit)]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or item.get("snippet") or "").strip()
        if content:
            content = content.replace("\n", " ").strip()
        payload = {
            "title": title,
            "url": url,
            "summary": _clamp_text(content, 280),
        }
        source = (item.get("source") or "").strip()
        if source:
            payload["source"] = source
        out.append(payload)
    return out


def _sec_query(section: str, fallback: str) -> str:
    if section == "business":
        return "business model, segments, products, customers, revenue drivers"
    if section == "risk":
        return "risk factors, competition, regulation, liquidity, debt, margin pressure"
    if section == "mda":
        return "management discussion and analysis, outlook, trends, guidance, MD&A"
    return fallback


def _sec_snippet_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    meta = chunk.get("metadata") or {}
    content = (chunk.get("content") or "").replace("\n", " ").strip()
    content = _clamp_text(content, MAX_SNIPPET_CHARS)
    payload = {
        "form_type": meta.get("form_type"),
        "filed_date": meta.get("filed_date"),
        "snippet": content,
    }
    return payload


def _history_as_text(history: List[Dict[str, str]], max_items: int) -> str:
    lines: List[str] = []
    for msg in history[-max_items:]:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def tool_get_user_profile(_args: NoArgs, ctx: ToolContext) -> Dict[str, Any]:
    if ctx.db is None or ctx.user_id is None:
        return {}
    profile = (
        ctx.db.query(UserOnboardingProfile)
        .filter(UserOnboardingProfile.user_id == ctx.user_id)
        .first()
    )
    return _serialize_onboarding(profile)

async def tool_get_portfolio_context(args: PortfolioContextInput, ctx: ToolContext) -> Dict[str, Any]:
    if ctx.db is None or ctx.user_id is None or ctx.finnhub is None:
        return {
            "as_of": int(time.time()),
            "currency": (ctx.user_currency or "USD").upper(),
            "portfolio_summary": {},
            "holdings": [],
            "symbols_for_news": [],
            "risk_flags": [],
        }

    currency = (ctx.user_currency or "USD").upper()
    as_of = int(time.time())

    try:
        # 1) One shared holdings snapshot (avoid double pricing calls)
        holdings_payload = ctx.holdings_snapshot
        if holdings_payload is None:
            holdings_payload = await get_holdings_with_live_prices(
                user_id=str(ctx.user_id),
                db=ctx.db,
                finnhub=ctx.finnhub,
                currency=currency,
                top_only=False,
                top_n=0,
                include_weights=True,
            )

        # 2) Summary from the same snapshot
        summary = await get_portfolio_summary(
            user_id=str(ctx.user_id),
            db=ctx.db,
            finnhub=ctx.finnhub,
            currency=currency,
            top_n=args.top_n,
            holdings_payload=holdings_payload,
        ) or {}

        # 3) Holdings list (brief)
        items = (holdings_payload or {}).get("items") or []
        holdings = [_holding_brief(it) for it in items if it]
        holdings = [h for h in holdings if h.get("symbol")]
        holdings = holdings[: args.max_holdings]

        # 4) Best symbols_for_news: from top_positions in summary
        symbols_for_news: List[str] = []
        top_positions = summary.get("top_positions") or []
        if isinstance(top_positions, list):
            for pos in top_positions:
                if not isinstance(pos, dict):
                    continue
                sym = (pos.get("symbol") or "").strip().upper()
                if sym and sym not in symbols_for_news:
                    symbols_for_news.append(sym)
                if len(symbols_for_news) >= 3:
                    break

        # Fallback: derive from holdings
        if not symbols_for_news:
            for h in holdings:
                sym = (h.get("symbol") or "").strip().upper()
                if sym and sym not in symbols_for_news:
                    symbols_for_news.append(sym)
                if len(symbols_for_news) >= 3:
                    break

        # 5) Optional quick risk flags (light heuristics; safe defaults)
        risk_flags: List[str] = []
        try:
            # if top_positions includes weights like {"weight_pct": ...}
            if top_positions and isinstance(top_positions, list):
                w1 = None
                w5 = 0.0
                for i, pos in enumerate(top_positions[:5]):
                    w = pos.get("weight_pct") or pos.get("weight") or None
                    if w is None:
                        continue
                    w = float(w)
                    if i == 0:
                        w1 = w
                    w5 += w
                if w1 is not None and w1 >= 35:
                    risk_flags.append("high_concentration_top1")
                if w5 >= 80:
                    risk_flags.append("high_concentration_top5")
        except Exception:
            pass

        return {
            "as_of": int(summary.get("as_of") or as_of),
            "currency": (summary.get("currency") or currency).upper(),
            "portfolio_summary": summary,
            "holdings": holdings,
            "symbols_for_news": symbols_for_news,
            "risk_flags": risk_flags,
        }

    except Exception as exc:
        logger.warning("tool_get_portfolio_context failed: %s", exc)
        return {
            "as_of": as_of,
            "currency": currency,
            "portfolio_summary": {},
            "holdings": [],
            "symbols_for_news": [],
            "risk_flags": [],
        }

async def tool_get_portfolio_summary(args: PortfolioSummaryInput, ctx: ToolContext) -> Dict[str, Any]:
    if ctx.db is None or ctx.user_id is None or ctx.finnhub is None:
        return {}
    try:
        return await get_portfolio_summary(
            user_id=str(ctx.user_id),
            db=ctx.db,
            finnhub=ctx.finnhub,
            currency=ctx.user_currency or "USD",
            top_n=args.top_n,
            holdings_payload=ctx.holdings_snapshot,
        )
    except Exception as exc:
        logger.warning("tool_get_portfolio_summary failed: %s", exc)
        return {}


async def tool_get_holdings(args: HoldingsInput, ctx: ToolContext) -> List[Dict[str, Any]]:
    if ctx.db is None or ctx.user_id is None or ctx.finnhub is None:
        return []
    try:
        payload = ctx.holdings_snapshot
        if payload is None:
            payload = await get_holdings_with_live_prices(
                user_id=str(ctx.user_id),
                db=ctx.db,
                finnhub=ctx.finnhub,
                currency=ctx.user_currency or "USD",
                top_only=False,
                top_n=0,
                include_weights=True,
            )
        items = (payload or {}).get("items") or []
        holdings = [_holding_brief(it) for it in items if it]
        holdings = [h for h in holdings if h.get("symbol")]
        return holdings[: args.max_items]
    except Exception as exc:
        logger.warning("tool_get_holdings failed: %s", exc)
        return []


async def tool_get_fundamentals(args: FundamentalsInput, _ctx: ToolContext) -> Dict[str, Any]:
    symbols = _normalize_symbols(args.symbols)
    if not symbols:
        return {"fundamentals": {}, "gaps": {}}
    tasks = [fetch_fundamentals_cached(sym, timeout_s=5.0) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    fundamentals: Dict[str, Any] = {}
    gaps: Dict[str, List[str]] = {}
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            gaps[sym] = [str(res)]
            continue
        fundamentals[sym] = res.data
        gaps[sym] = res.gaps
    return {"fundamentals": fundamentals, "gaps": gaps}


async def tool_get_sec_snippets(args: SecSnippetsInput, ctx: ToolContext) -> Dict[str, Any]:
    symbol = (args.symbol or "").strip().upper()
    if not symbol or ctx.db is None:
        return {"symbol": symbol, "section": args.section, "snippets": []}
    section = args.section if args.section in ALLOWED_SEC_SECTIONS else "general"
    query = _sec_query(section, args.query or ctx.message or "")
    if not query:
        return {"symbol": symbol, "section": section, "snippets": []}
    vector_service = VectorStoreService()
    try:
        chunks = vector_service.get_context_for_analysis(
            db=ctx.db,
            symbol=symbol,
            query=query,
            limit=args.limit,
        )
        snippets = []
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            snippet = _sec_snippet_payload(chunk)
            snippet["symbol"] = symbol
            snippets.append(snippet)
        return {"symbol": symbol, "section": section, "snippets": snippets}
    except Exception as exc:
        logger.warning("tool_get_sec_snippets failed: %s", exc)
        return {"symbol": symbol, "section": section, "snippets": []}


async def tool_get_news(args: NewsInput, ctx: ToolContext) -> Dict[str, Any]:
    # 1) Normalize
    symbols = [s.upper().strip() for s in (args.symbols or []) if isinstance(s, str) and s.strip()]
    if not symbols:
        symbols = [s.upper().strip() for s in (ctx.symbols or []) if isinstance(s, str) and s.strip()]

    topic = (args.topic or "").strip()
    query = (args.query or "").strip()

    # 2) Build query (priority: explicit query > symbols > topic > fallback)
    if query:
        final_query = query
    elif symbols:
        # if you pass multiple tickers, pick first 1–2 to keep Tavily query focused
        pick = symbols[:2]
        if len(pick) == 1:
            final_query = f"{pick[0]} latest news"
        else:
            final_query = f"{pick[0]} {pick[1]} latest news"
    elif topic:
        final_query = f"latest financial news {topic}"
    else:
        # last resort: still allow message, but keep it short and safe
        msg = (ctx.message or "").strip()
        final_query = msg if len(msg) >= 3 else "latest financial markets news"

    final_query = _clamp_text(final_query, MAX_QUERY_CHARS)
    if not final_query:
        return {"query": "", "items": []}

    # 3) Cache by query (good)
    cache_key = _news_cache_key(final_query)
    cached = cache_get(cache_key)
    if isinstance(cached, list):
        return {"query": final_query, "items": cached[: args.max_results]}

    # 4) Search
    try:
        results = await tavily_search(
            query=final_query,
            max_results=args.max_results,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
            days=args.days,  # use args.days instead of constant
        )
        items = _normalize_news_results(results, args.max_results)
        cache_set(cache_key, items, ttl_seconds=TTL_TAVILY_SEC)
        return {"query": final_query, "items": items}
    except Exception as exc:
        logger.warning("tool_get_news failed: %s", exc)
        return {"query": final_query, "items": []}



async def tool_get_web_search(args: WebSearchInput, ctx: ToolContext) -> Dict[str, Any]:
    query = (args.query or "").strip()
    if not query:
        if ctx.symbols:
            query = f"{ctx.symbols[0]} overview"
        else:
            query = ctx.message or ""
    query = _clamp_text(query, MAX_QUERY_CHARS)
    if not query:
        return {"query": "", "items": []}
    cache_key = _web_search_cache_key(query)
    cached = cache_get(cache_key)
    if isinstance(cached, list):
        return {"query": query, "items": cached[: args.max_results]}
    try:
        results = await tavily_search(
            query=query,
            max_results=args.max_results,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
            include_domains=TAVILY_WEB_DOMAINS or None,
            days=TAVILY_WEB_SEARCH_DAYS,
        )
        items = _normalize_search_results(results, args.max_results)
        cache_set(cache_key, items, ttl_seconds=TTL_WEB_SEARCH_SEC)
        return {"query": query, "items": items}
    except Exception as exc:
        logger.warning("tool_get_web_search failed: %s", exc)
        return {"query": query, "items": []}


async def tool_get_chat_history(args: ChatHistoryInput, ctx: ToolContext) -> Dict[str, Any]:
    history = ctx.history or []
    max_items = _clamp_int(args.max_items, 10, 1, 50)
    trimmed = history[-max_items:] if history else []
    return {
        "messages": trimmed,
        "text": _history_as_text(trimmed, max_items),
    }


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "get_user_profile": ToolSpec(
        name="get_user_profile",
        description="Fetches the user's onboarding profile for personalization.",
        input_model=NoArgs,
        run=tool_get_user_profile,
    ),
    "get_portfolio_context": ToolSpec(
        name="get_portfolio_context",
        description="Fetches comprehensive portfolio context including summary, holdings, and symbols for news.",
        input_model=PortfolioContextInput,
        run=tool_get_portfolio_context,
    ),
    "get_fundamentals": ToolSpec(
        name="get_fundamentals",
        description="Fetches fundamentals for explicit ticker symbols.",
        input_model=FundamentalsInput,
        run=tool_get_fundamentals,
    ),
    "get_sec_snippets": ToolSpec(
        name="get_sec_snippets",
        description="Fetches short SEC filing snippets for a symbol and section.",
        input_model=SecSnippetsInput,
        run=tool_get_sec_snippets,
    ),
    "get_news": ToolSpec(
        name="get_news",
        description="Fetches recent finance news for a query.",
        input_model=NewsInput,
        run=tool_get_news,
    ),
    "get_web_search": ToolSpec(
        name="get_web_search",
        description="Searches the web for general financial context beyond internal data.",
        input_model=WebSearchInput,
        run=tool_get_web_search,
    ),
    "get_chat_history": ToolSpec(
        name="get_chat_history",
        description="Fetches recent chat history for context or recap.",
        input_model=ChatHistoryInput,
        run=tool_get_chat_history,
    ),
}


def render_tool_manifest(tool_names: List[str]) -> str:
    lines: List[str] = []
    for name in tool_names:
        spec = TOOL_REGISTRY.get(name)
        if not spec:
            continue
        schema = spec.input_model.model_json_schema()
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        args = []
        for key, info in props.items():
            arg_type = info.get("type", "any")
            if "enum" in info:
                arg_type = f"{arg_type} enum={info['enum']}"
            if key in required:
                arg_type = f"{arg_type} required"
            args.append(f"{key}: {arg_type}")
        arg_text = ", ".join(args) if args else "no args"
        lines.append(f"- {spec.name}: {spec.description} Args: {arg_text}")
    return "\n".join(lines)


def prepare_tool_args(
    tool_name: str,
    raw_args: Dict[str, Any],
    caps: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    args = dict(raw_args or {})
    if tool_name == "get_user_profile":
        return {}
    if tool_name == "get_portfolio_context":
        top_n_cap = _clamp_int(caps.get("top_n"), 8, 1, 15)
        max_holdings_cap = _clamp_int(caps.get("max_holdings"), 25, 5, 100)
        return {
            "top_n": _clamp_int(args.get("top_n"), top_n_cap,1, top_n_cap),
            "max_holdings": _clamp_int(args.get("max_holdings"), max_holdings_cap, 5, max_holdings_cap),
        }
    if tool_name == "get_fundamentals":
        symbols = _normalize_symbols(args.get("symbols") or state.get("symbols") or [])
        max_symbols = _clamp_int(caps.get("max_symbols"), 3, 1, 10)
        return {"symbols": symbols[:max_symbols]}
    if tool_name == "get_sec_snippets":
        symbols = _normalize_symbols(state.get("symbols") or [])
        symbol = (args.get("symbol") or (symbols[0] if symbols else "")).strip().upper()
        section = (args.get("section") or "general").strip().lower()
        if section not in ALLOWED_SEC_SECTIONS:
            section = "general"
        query = (args.get("query") or state.get("message") or "").strip()
        limit_cap = _clamp_int(caps.get("max_snippets"), 6, 1, 10)
        limit = _clamp_int(args.get("limit"), limit_cap, 1, limit_cap)
        return {
            "symbol": symbol,
            "section": section,
            "query": _clamp_text(query, MAX_QUERY_CHARS),
            "limit": limit,
        }
    if tool_name == "get_news":
        query = (args.get("query") or "").strip()
        if not query:
            symbols = _normalize_symbols(state.get("symbols") or [])
            if symbols:
                query = f"latest news for {symbols[0]}"
            else:
                query = (state.get("message") or "").strip()
        max_results_cap = _clamp_int(caps.get("max_results"), MAX_NEWS_RESULTS, 1, 10)
        max_results = _clamp_int(args.get("max_results"), max_results_cap, 1, max_results_cap)
        return {
            "query": _clamp_text(query, MAX_QUERY_CHARS),
            "max_results": max_results,
        }
    if tool_name == "get_web_search":
        query = (args.get("query") or "").strip()
        if not query:
            query = (state.get("message") or "").strip()
        max_results_cap = _clamp_int(caps.get("max_results"), 5, 1, 10)
        max_results = _clamp_int(args.get("max_results"), max_results_cap, 1, max_results_cap)
        return {
            "query": _clamp_text(query, MAX_QUERY_CHARS),
            "max_results": max_results,
        }
    if tool_name == "get_chat_history":
        max_items_cap = _clamp_int(caps.get("max_items"), 10, 1, 50)
        return {"max_items": _clamp_int(args.get("max_items"), max_items_cap, 1, max_items_cap)}
    return args


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> BaseModel | None:
    spec = TOOL_REGISTRY.get(tool_name)
    if not spec:
        return None
    try:
        return spec.input_model.model_validate(args)
    except ValidationError:
        return None
