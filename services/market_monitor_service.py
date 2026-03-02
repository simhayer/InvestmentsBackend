from __future__ import annotations

import asyncio
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Iterable

from services.ai.llm_service import get_llm_service
from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.finnhub_news_service import (
    get_company_news_for_symbols,
    get_global_news,
    get_global_news_cached,
)
from services.global_brief_service import get_global_brief_cached, get_predictions_cached

logger = logging.getLogger(__name__)

CACHE_KEY_MONITOR_PANEL = "market:monitor_panel:v1"
TTL_MONITOR_PANEL_SEC = int(os.getenv("TTL_MONITOR_PANEL_SEC", "600"))

AI_INSIGHTS_SYSTEM_PROMPT = """You are generating compact AI insight cards for a global finance dashboard.

Return valid JSON only with this schema:
{
  "cards": [
    {
      "title": "Short title",
      "summary": "One or two direct sentences.",
      "signal": "bullish|bearish|neutral",
      "time_horizon": "intraday|1-3d|1-2w"
    }
  ]
}

Rules:
- Return 2-3 cards.
- Ground every card in the provided brief, outlook, market pulse, and news themes.
- Keep titles short and concrete.
- Do not use markdown.
"""

_MARKET_PULSE_GROUPS = (
    ("major_indices", "Major Indices", {"SPX", "DJI", "IXIC", "RUT"}),
    ("crypto", "Crypto", {"BTC", "ETH"}),
    ("risk_signals", "Risk Signals", {"VIX", "GLD"}),
)

_NEWS_STREAMS = (
    ("general", "Top Stories"),
    ("merger", "Deal Flow"),
    ("forex", "FX Watch"),
    ("crypto", "Crypto Watch"),
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pick_as_of(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return _iso_now()


def _cache_key_personalized(user_id: str, currency: str, symbols: Iterable[str], watchlist_id: int | None = None) -> str:
    normalized = ",".join(sorted({(symbol or "").strip().upper() for symbol in symbols if symbol}))
    base = f"{watchlist_id or 'portfolio'}:{normalized}"
    suffix = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12] if base else "portfolio"
    return f"market:monitor_panel:personalized:{user_id}:{currency.upper()}:{suffix}"


def _compact_news_items(items: List[Dict[str, Any]], *, limit: int = 4) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for item in items[:limit]:
        compact.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "snippet": item.get("snippet"),
                "image": item.get("image"),
            }
        )
    return compact


def _normalize_symbols(symbols: List[str] | str | None, *, limit: int = 8) -> List[str]:
    if symbols is None:
        return []
    raw_items = symbols if isinstance(symbols, list) else str(symbols).split(",")
    out: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        sym = (item or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= limit:
            break
    return out


def _build_news_streams(news_by_category: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    for key, label in _NEWS_STREAMS:
        items = news_by_category.get(key) or []
        streams.append(
            {
                "key": key,
                "label": label,
                "items": _compact_news_items(items),
            }
        )
    return streams


def _build_market_pulse(overview: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = overview.get("items") or []
    by_key = {str(item.get("key") or ""): item for item in items}

    groups: List[Dict[str, Any]] = []
    for key, label, item_keys in _MARKET_PULSE_GROUPS:
        group_items = [by_key[item_key] for item_key in item_keys if item_key in by_key]
        groups.append(
            {
                "key": key,
                "label": label,
                "items": group_items,
            }
        )
    return groups


def _compact_position(position: Any) -> Dict[str, Any]:
    getter = position.get if isinstance(position, dict) else lambda key, default=None: getattr(position, key, default)
    return {
        "symbol": getter("symbol"),
        "name": getter("name"),
        "weight": getter("weight"),
        "current_value": getter("current_value"),
        "unrealized_pl_pct": getter("unrealized_pl_pct"),
        "current_price": getter("current_price"),
        "currency": getter("currency"),
    }


def _infer_signal(*texts: str) -> str:
    text = " ".join(t.lower() for t in texts if t)
    bearish_terms = ("risk-off", "pressure", "selloff", "bearish", "weaker", "slowdown", "inflation", "sanction")
    bullish_terms = ("risk-on", "upside", "bullish", "supportive", "rebound", "easing", "stronger")

    if any(term in text for term in bearish_terms):
        return "bearish"
    if any(term in text for term in bullish_terms):
        return "bullish"
    return "neutral"


def _fallback_ai_insights(
    brief: Dict[str, Any],
    predictions: Dict[str, Any],
    market_pulse: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    if predictions.get("outlook"):
        cards.append(
            {
                "title": "Market Regime",
                "summary": predictions["outlook"],
                "signal": _infer_signal(predictions["outlook"]),
                "time_horizon": "1-3d",
            }
        )

    for section in (brief.get("sections") or [])[:2]:
        title = (section.get("headline") or "Market Theme").strip()
        summary = (section.get("impact") or section.get("cause") or "").strip()
        if not summary:
            continue
        cards.append(
            {
                "title": title,
                "summary": summary,
                "signal": _infer_signal(section.get("cause") or "", section.get("impact") or ""),
                "time_horizon": "1-3d",
            }
        )

    if len(cards) < 3:
        for group in market_pulse:
            items = group.get("items") or []
            movers = [item for item in items if item.get("changePct") is not None]
            if not movers:
                continue
            movers.sort(key=lambda item: abs(float(item.get("changePct") or 0)), reverse=True)
            top = movers[0]
            direction = "higher" if float(top.get("changePct") or 0) >= 0 else "lower"
            cards.append(
                {
                    "title": group.get("label") or "Market Pulse",
                    "summary": f'{top.get("label")} is trading {direction} at {top.get("changePct"):.2f}% and is the clearest move in this block.',
                    "signal": "bullish" if direction == "higher" else "bearish",
                    "time_horizon": "intraday",
                }
            )
            if len(cards) >= 3:
                break

    return cards[:3]


def _build_personalized_insight_cards(
    inline_insights: Dict[str, Any] | None,
    *,
    focus_scope: str,
    symbols: List[str],
    symbol_news: Dict[str, List[Dict[str, Any]]] | None = None,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    if inline_insights:
        mapping = [
            ("Portfolio Health", inline_insights.get("healthBadge"), "1-2w"),
            ("Performance", inline_insights.get("performanceNote"), "1-2w"),
            ("Risk Flag", inline_insights.get("riskFlag"), "1-2w"),
            ("Action To Review", inline_insights.get("actionNeeded"), "1-2w"),
            ("Top Performer", inline_insights.get("topPerformer"), "1-2w"),
        ]
        for title, summary, horizon in mapping:
            if not summary:
                continue
            cards.append(
                {
                    "title": title,
                    "summary": summary,
                    "signal": _infer_signal(str(summary)),
                    "time_horizon": horizon,
                }
            )
            if len(cards) >= 4:
                return cards

    if focus_scope == "watchlist":
        for symbol in symbols[:3]:
            items = (symbol_news or {}).get(symbol) or []
            if not items:
                continue
            lead = items[0]
            summary = (lead.get("title") or "").strip()
            snippet = (lead.get("snippet") or "").strip()
            cards.append(
                {
                    "title": f"{symbol} Watch",
                    "summary": f"{summary}. {snippet}".strip(),
                    "signal": _infer_signal(summary, snippet),
                    "time_horizon": "1-3d",
                }
            )
        if cards:
            return cards[:4]

    return cards[:4]


def _build_ai_insight_prompt(
    brief: Dict[str, Any],
    predictions: Dict[str, Any],
    market_pulse: List[Dict[str, Any]],
    news_streams: List[Dict[str, Any]],
) -> str:
    pulse_lines: List[str] = []
    for group in market_pulse:
        pulse_lines.append(f"- {group.get('label')}")
        for item in (group.get("items") or [])[:4]:
            pulse_lines.append(
                f"  - {item.get('label')}: price={item.get('price')} changePct={item.get('changePct')} currency={item.get('currency')}"
            )

    news_lines: List[str] = []
    for stream in news_streams:
        news_lines.append(f"- {stream.get('label')}")
        for item in (stream.get("items") or [])[:3]:
            news_lines.append(f"  - {item.get('title')} [{item.get('source')}]")

    return "\n".join(
        [
            f"Outlook: {predictions.get('outlook') or brief.get('outlook') or 'Unavailable'}",
            "",
            "Brief sections:",
            *[
                f"- {section.get('headline')}: {section.get('cause')} | Impact: {section.get('impact')}"
                for section in (brief.get("sections") or [])[:4]
            ],
            "",
            "Market pulse:",
            *pulse_lines,
            "",
            "News streams:",
            *news_lines,
        ]
    )


async def _generate_ai_insights(
    brief: Dict[str, Any],
    predictions: Dict[str, Any],
    market_pulse: List[Dict[str, Any]],
    news_streams: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:
        llm = get_llm_service()
        raw = await llm.generate_json(
            system=AI_INSIGHTS_SYSTEM_PROMPT,
            user=_build_ai_insight_prompt(brief, predictions, market_pulse, news_streams),
        )
        cards = raw.get("cards") or []
        cleaned: List[Dict[str, Any]] = []
        for card in cards[:3]:
            if not isinstance(card, dict):
                continue
            title = (card.get("title") or "").strip()
            summary = (card.get("summary") or "").strip()
            if not title or not summary:
                continue
            signal = (card.get("signal") or "neutral").strip().lower()
            if signal not in {"bullish", "bearish", "neutral"}:
                signal = "neutral"
            horizon = (card.get("time_horizon") or "1-3d").strip()
            cleaned.append(
                {
                    "title": title,
                    "summary": summary,
                    "signal": signal,
                    "time_horizon": horizon,
                }
            )
        if cleaned:
            return cleaned
    except Exception as exc:
        logger.exception("market_monitor_ai_insights_failed: %s", exc)

    return _fallback_ai_insights(brief, predictions, market_pulse)


async def _fetch_news_bundle(force_refresh: bool) -> Dict[str, List[Dict[str, Any]]]:
    async def _fetch_one(category: str) -> List[Dict[str, Any]]:
        if force_refresh:
            return await get_global_news(category=category, limit=6)
        return await get_global_news_cached(category=category, limit=6)

    categories = [key for key, _label in _NEWS_STREAMS]
    results = await asyncio.gather(*[_fetch_one(category) for category in categories])
    return dict(zip(categories, results))


def _get_market_overview_data(db: Any, *, force_refresh: bool) -> Dict[str, Any]:
    from services.market_service import get_market_overview_cached

    return get_market_overview_cached(db, max_age_sec=0 if force_refresh else 60)


async def _get_live_holdings_payload(
    user_id: str,
    db: Any,
    finnhub: Any,
    *,
    currency: str,
    top_n: int,
) -> Dict[str, Any]:
    from services.holding_service import get_holdings_with_live_prices

    return await get_holdings_with_live_prices(
        user_id,
        db,
        finnhub,
        currency=currency,
        top_only=False,
        top_n=top_n,
        include_weights=True,
    )


async def _get_portfolio_summary_data(
    user_id: str,
    db: Any,
    finnhub: Any,
    *,
    currency: str,
    holdings_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from services.portfolio.portfolio_service import get_portfolio_summary

    return await get_portfolio_summary(
        user_id,
        db,
        finnhub,
        currency=currency,
        holdings_payload=holdings_payload,
    )


async def _get_portfolio_inline_insights_data(
    user_id: str,
    db: Any,
    finnhub: Any,
    *,
    currency: str,
    force_refresh: bool,
) -> Dict[str, Any]:
    from services.ai.portfolio.analyze_portfolio_service import get_portfolio_insights

    return await get_portfolio_insights(
        user_id,
        db,
        finnhub,
        currency=currency,
        force_refresh=force_refresh,
    )


def _get_watchlist_context(db: Any, user_id: str, watchlist_id: int) -> tuple[Dict[str, Any], list[str]]:
    from services.watchlist_service import get_watchlist_symbols

    watchlist, symbols = get_watchlist_symbols(db, int(user_id), watchlist_id)
    return {
        "id": watchlist.id,
        "name": watchlist.name,
        "is_default": watchlist.is_default,
    }, symbols


async def get_market_monitor_panel(db: Any, *, force_refresh: bool = False) -> Dict[str, Any]:
    if not force_refresh:
        cached = cache_get(CACHE_KEY_MONITOR_PANEL)
        if isinstance(cached, dict) and isinstance(cached.get("sections"), dict):
            return cached

    brief_task = get_global_brief_cached(force_refresh=force_refresh)
    predictions_task = get_predictions_cached(force_refresh=force_refresh)
    news_task = _fetch_news_bundle(force_refresh)
    overview_task = asyncio.to_thread(
        _get_market_overview_data,
        db,
        force_refresh=force_refresh,
    )

    brief, predictions, news_by_category, overview = await asyncio.gather(
        brief_task,
        predictions_task,
        news_task,
        overview_task,
    )

    market_pulse = _build_market_pulse(overview)
    news_streams = _build_news_streams(news_by_category)
    ai_insights = await _generate_ai_insights(brief, predictions, market_pulse, news_streams)

    payload = {
        "as_of": _pick_as_of(
            predictions.get("as_of"),
            brief.get("as_of"),
            overview.get("fetched_at"),
        ),
        "title": "Global Finance Monitor",
        "subtitle": "World brief, AI signals, market pulse, and themed news streams for the left panel.",
        "outlook": predictions.get("outlook") or brief.get("outlook"),
        "sections": {
            "world_brief": {
                "market": brief.get("market") or "Global Markets",
                "sections": brief.get("sections") or [],
            },
            "ai_insights": ai_insights,
            "market_pulse": market_pulse,
            "news_streams": news_streams,
        },
        "meta": {
            "sources": ["global_brief", "predictions", "market_overview", "global_news"],
            "generated_at": _iso_now(),
            "news_categories": [key for key, _label in _NEWS_STREAMS],
        },
    }

    cache_set(CACHE_KEY_MONITOR_PANEL, payload, ttl_seconds=TTL_MONITOR_PANEL_SEC)
    return payload


async def get_personalized_market_monitor_panel(
    db: Any,
    *,
    user_id: str,
    finnhub: Any,
    currency: str,
    force_refresh: bool = False,
    watchlist_id: int | None = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    watchlist_ref: Dict[str, Any] | None = None
    if watchlist_id is not None:
        watchlist_ref, watchlist_symbols = _get_watchlist_context(db, user_id, watchlist_id)
        focus_symbols = _normalize_symbols(watchlist_symbols, limit=top_n)
    else:
        focus_symbols = []

    personalized_cache_key = _cache_key_personalized(
        user_id,
        currency,
        focus_symbols,
        watchlist_id=watchlist_id,
    )

    if not force_refresh:
        cached = cache_get(personalized_cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("personalization"), dict):
            return cached

    base_panel = await get_market_monitor_panel(db, force_refresh=force_refresh)

    top_positions: List[Dict[str, Any]] = []
    portfolio_snapshot: Dict[str, Any] | None = None
    inline_insights: Dict[str, Any] | None = None
    scope = "watchlist" if watchlist_id is not None else "portfolio"

    if watchlist_id is None:
        holdings_payload = await _get_live_holdings_payload(
            user_id,
            db,
            finnhub,
            currency=currency,
            top_n=top_n,
        )
        raw_top_positions = (holdings_payload.get("top_items") or [])[:top_n]
        top_positions = [_compact_position(position) for position in raw_top_positions]
        focus_symbols = _normalize_symbols([position.get("symbol") for position in top_positions], limit=top_n)

        if focus_symbols:
            portfolio_snapshot = await _get_portfolio_summary_data(
                user_id,
                db,
                finnhub,
                currency=currency,
                holdings_payload=holdings_payload,
            )
            inline_insights = await _get_portfolio_inline_insights_data(
                user_id,
                db,
                finnhub,
                currency=currency,
                force_refresh=force_refresh,
            )
        else:
            scope = "global_fallback"
    elif not focus_symbols:
        scope = "watchlist"

    symbol_news = (
        await get_company_news_for_symbols(
            focus_symbols,
            days_back=7,
            limit_per_symbol=3,
        )
        if focus_symbols
        else {}
    )

    focus_news = [
        {
            "symbol": symbol,
            "items": _compact_news_items(symbol_news.get(symbol) or [], limit=3),
        }
        for symbol in focus_symbols
    ]

    personalized_cards = _build_personalized_insight_cards(
        inline_insights,
        focus_scope=scope,
        symbols=focus_symbols,
        symbol_news=symbol_news,
    )

    payload = dict(base_panel)
    payload["personalization"] = {
        "scope": scope,
        "currency": currency.upper(),
        "symbols": focus_symbols,
        "watchlist": watchlist_ref,
        "top_positions": top_positions,
        "portfolio_snapshot": portfolio_snapshot,
        "inline_insights": inline_insights,
        "insight_cards": personalized_cards,
        "focus_news": focus_news,
        "empty_state": (
            None
            if focus_symbols
            else "No holdings found. Create a watchlist or add holdings to personalize this panel."
        ),
    }

    cache_set(personalized_cache_key, payload, ttl_seconds=TTL_MONITOR_PANEL_SEC)
    return payload
