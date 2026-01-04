from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, TypedDict

from dateutil import parser

from services.news.filters import canonicalize_url, classify_source, extract_domain
from services.news.ranking import (
    best_source_tier,
    filter_domains,
    has_event_keyword,
    rank_key,
    score_item,
)
from services.tavily.client import TavilyClientError, search as tavily_search

NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", os.getenv("SINGLE_STOCK_NEWS_MAX_ITEMS", "8")))
NEWS_SNIPPET_MAX_CHARS = int(os.getenv("NEWS_SNIPPET_MAX_CHARS", "500"))
NEWS_RECENCY_DAYS = int(os.getenv("NEWS_RECENCY_DAYS", "14"))
NEWS_MIN_RECENT_ITEMS = int(os.getenv("NEWS_MIN_RECENT_ITEMS", "3"))

DATE_PATTERNS = [
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\b", re.IGNORECASE),
]

URL_DATE_PATTERNS = [
    re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/"),
    re.compile(r"/(20\d{2})-(\d{2})-(\d{2})/"),
]


class NewsItem(TypedDict):
    id: str
    title: str
    source_domain: str
    published_at: Optional[str]
    url: str
    snippet: str
    source_tier: str


@dataclass(frozen=True)
class NewsFetchResult:
    items: List[NewsItem]
    data_gaps: List[str]
    recency_days_used: int


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parser.parse(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_from_text(text: str) -> Optional[datetime]:
    if not text:
        return None
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _parse_datetime(match.group(0))
    return None


def _parse_from_url(url: str) -> Optional[datetime]:
    for pattern in URL_DATE_PATTERNS:
        match = pattern.search(url)
        if match:
            year, month, day = match.groups()
            return _parse_datetime(f"{year}-{month}-{day}")
    return None


def extract_published_at(result: dict[str, Any]) -> Optional[datetime]:
    raw = _normalize_text(
        result.get("published_date")
        or result.get("published")
        or result.get("date")
        or result.get("published_time")
        or ""
    )
    parsed = _parse_datetime(raw)
    if parsed:
        return parsed

    title = _normalize_text(result.get("title") or result.get("headline") or "")
    snippet = _normalize_text(result.get("content") or result.get("snippet") or "")
    url = _normalize_text(result.get("url") or result.get("link") or "")

    parsed = _parse_from_text(snippet)
    if parsed:
        return parsed
    parsed = _parse_from_text(title)
    if parsed:
        return parsed
    return _parse_from_url(url)


def _candidate_items(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for order, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        title = _normalize_text(item.get("title") or item.get("headline") or "")
        url = _normalize_text(item.get("url") or item.get("link") or "")
        if not title or not url:
            continue
        snippet = _normalize_text(
            item.get("content")
            or item.get("snippet")
            or item.get("description")
            or ""
        )
        if not snippet:
            snippet = title
        canonical_url = canonicalize_url(url)
        classification = classify_source(canonical_url, title)
        published_at = extract_published_at(item)
        event_match = has_event_keyword(title) or has_event_keyword(snippet)
        parsed.append(
            {
                "title": title,
                "url": canonical_url,
                "source_domain": extract_domain(canonical_url),
                "published_at": published_at,
                "snippet": _truncate(snippet, NEWS_SNIPPET_MAX_CHARS),
                "classification": classification,
                "order": order,
                "event_match": event_match,
            }
        )
    return parsed


def _dedupe_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = (item.get("url") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _filter_recency(
    items: Iterable[dict[str, Any]],
    *,
    recency_days: int,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    kept: list[dict[str, Any]] = []
    for item in items:
        classification = item.get("classification")
        if classification in {"quote_junk", "options_junk"}:
            continue
        published_at: Optional[datetime] = item.get("published_at")
        if published_at:
            age_days = (now - published_at).total_seconds() / 86400.0
            if age_days <= recency_days:
                item["age_days"] = age_days
                kept.append(item)
            continue
        if classification in {"tier1_news", "press_release", "filing"} and item.get("event_match"):
            item["age_days"] = None
            kept.append(item)
    return kept


def normalize_news_results(
    results: Iterable[dict[str, Any]],
    *,
    max_items: int = NEWS_MAX_ITEMS,
    recency_days: int = NEWS_RECENCY_DAYS,
    min_recent_items: int = NEWS_MIN_RECENT_ITEMS,
) -> NewsFetchResult:
    data_gaps: list[str] = []
    candidates = _candidate_items(results)
    candidates = _dedupe_items(candidates)

    kept = _filter_recency(candidates, recency_days=recency_days)
    used_recency_days = recency_days
    if len(kept) < min_recent_items:
        expanded_days = max(recency_days, 30)
        data_gaps.append(
            f"Only {len(kept)} items found within {recency_days} days; expanded window to {expanded_days} days."
        )
        kept = _filter_recency(candidates, recency_days=expanded_days)
        used_recency_days = expanded_days

    if len(kept) < min_recent_items:
        data_gaps.append(
            f"Insufficient recent high-quality news found ({len(kept)} items)."
        )

    ranked: list[dict[str, Any]] = []
    for item in kept:
        source_domain = item.get("source_domain", "")
        source_tier = best_source_tier(source_domain)
        classification = item.get("classification", "analysis_low")
        score = score_item(
            title=item.get("title", ""),
            source_domain=source_domain,
            source_tier=source_tier,
            classification=classification,
            published_at=item.get("published_at"),
            recency_days=used_recency_days,
        )
        ranked.append({**item, "source_tier": source_tier, "score": score})

    ranked.sort(
        key=lambda row: rank_key(
            score=row.get("score", 0.0),
            published_at=row.get("published_at"),
            order=row.get("order", 0),
        ),
        reverse=True,
    )

    out: list[NewsItem] = []
    for idx, item in enumerate(ranked[: max(0, int(max_items))], start=1):
        published_at = item.get("published_at")
        published_at_str = published_at.isoformat() if isinstance(published_at, datetime) else None
        source_tier = item.get("source_tier", "low_signal")
        out.append(
            {
                "id": f"news_{idx}",
                "title": item.get("title", ""),
                "source_domain": item.get("source_domain", ""),
                "published_at": published_at_str,
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source_tier": "tier1" if source_tier == "tier1" else source_tier,
            }
        )
    return NewsFetchResult(items=out, data_gaps=data_gaps, recency_days_used=used_recency_days)


async def fetch_news(symbol: str, company_name: Optional[str] = None) -> NewsFetchResult:
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        return NewsFetchResult(
            items=[],
            data_gaps=["Missing symbol for news query"],
            recency_days_used=NEWS_RECENCY_DAYS,
        )

    primary = f"{clean_symbol} stock news last 7 days"
    secondary = None
    if company_name:
        clean_name = " ".join(company_name.strip().split())
        if clean_name:
            secondary = f"{clean_name} ({clean_symbol}) latest news last 7 days"
    tertiary = f"{clean_symbol} earnings guidance news last 30 days"

    preferred_domains = filter_domains(
        [
            "reuters.com",
            "bloomberg.com",
            "cnbc.com",
            "ft.com",
            "wsj.com",
        ]
    )
    if preferred_domains:
        domain_clause = " OR ".join(f"site:{domain}" for domain in preferred_domains)
        targeted = (
            f"{clean_symbol} (earnings OR guidance OR acquisition OR partnership OR lawsuit "
            f"OR regulator OR antitrust OR merger) ({domain_clause})"
        )
    else:
        targeted = None

    queries = [primary, tertiary]
    if secondary:
        queries.insert(1, secondary)
    if targeted:
        queries.append(targeted)

    tasks = [
        tavily_search(
            query=q,
            max_results=NEWS_MAX_ITEMS * 2,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
        )
        for q in queries
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict[str, Any]] = []
    for resp in responses:
        if isinstance(resp, Exception):
            continue
        results = resp.get("results")
        if isinstance(results, list):
            combined.extend(results)

    try:
        return normalize_news_results(
            combined,
            max_items=NEWS_MAX_ITEMS,
            recency_days=NEWS_RECENCY_DAYS,
            min_recent_items=NEWS_MIN_RECENT_ITEMS,
        )
    except TavilyClientError:
        return NewsFetchResult(
            items=[],
            data_gaps=["News search failed"],
            recency_days_used=NEWS_RECENCY_DAYS,
        )
