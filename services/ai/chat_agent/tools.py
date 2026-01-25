from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal

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
    query: str | None = None
    max_results: int = Field(default=6, ge=1, le=10)


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
    query = (args.query or "").strip()
    if not query:
        if ctx.symbols:
            query = f"latest news for {ctx.symbols[0]}"
        else:
            query = ctx.message or ""
    query = _clamp_text(query, MAX_QUERY_CHARS)
    if not query:
        return {"query": "", "items": []}
    cache_key = _news_cache_key(query)
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
        )
        items = _normalize_news_results(results, args.max_results)
        cache_set(cache_key, items, ttl_seconds=TTL_TAVILY_SEC)
        return {"query": query, "items": items}
    except Exception as exc:
        logger.warning("tool_get_news failed: %s", exc)
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
    "get_portfolio_summary": ToolSpec(
        name="get_portfolio_summary",
        description="Fetches portfolio summary metrics and top holdings overview.",
        input_model=PortfolioSummaryInput,
        run=tool_get_portfolio_summary,
    ),
    "get_holdings": ToolSpec(
        name="get_holdings",
        description="Fetches the user's holdings with live prices and weights.",
        input_model=HoldingsInput,
        run=tool_get_holdings,
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
    if tool_name == "get_portfolio_summary":
        top_n_cap = _clamp_int(caps.get("top_n"), 5, 1, 10)
        return {"top_n": _clamp_int(args.get("top_n"), top_n_cap, 1, top_n_cap)}
    if tool_name == "get_holdings":
        max_items_cap = _clamp_int(caps.get("max_items"), 25, 1, 100)
        return {"max_items": _clamp_int(args.get("max_items"), max_items_cap, 1, max_items_cap)}
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
