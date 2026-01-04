"""Async single-stock analysis pipeline using Tavily + Finnhub + OpenAI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from schemas.stock_report import Citation, StockReport
from services.filings.tavily_filings import fetch_filings, needs_filings_for_request
from services.fundamentals.finnhub_fundamentals import FundamentalsResult, fetch_fundamentals
from services.news.tavily_news import fetch_news
from services.synthesis.stock_report_synthesizer import synthesize_stock_report

try:
    import redis as redis_sync  # redis-py for direct Redis/Upstash (redis:// / rediss://)
except Exception:
    redis_sync = None

SINGLE_STOCK_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_CACHE_TTL_SEC", "86400"))
LOCAL_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_LOCAL_CACHE_TTL_SEC", "60"))
SINGLE_STOCK_SCHEMA_VERSION = os.getenv("SINGLE_STOCK_SCHEMA_VERSION", "3")

NEWS_TIMEOUT_SEC = float(os.getenv("NEWS_TIMEOUT_SEC", "10"))
FILINGS_TIMEOUT_SEC = float(os.getenv("FILINGS_TIMEOUT_SEC", "10"))
FUNDAMENTALS_TIMEOUT_SEC = float(os.getenv("FUNDAMENTALS_TIMEOUT_SEC", "5"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "100"))
GLOBAL_TIMEOUT_SEC = float(os.getenv("GLOBAL_TIMEOUT", os.getenv("GLOBAL_TIMEOUT_SEC", "1000")))

_redis_client: Optional[Any] = None
_LOCAL_CACHE: dict[str, tuple[float, Any]] = {}
_MISS = object()
logger = logging.getLogger(__name__)


@dataclass
class TimeBudget:
    total_sec: float
    start: float = field(default_factory=time.perf_counter)

    def remaining(self) -> float:
        return max(0.0, self.total_sec - (time.perf_counter() - self.start))


def _get_redis_client() -> Optional[Any]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not redis_sync:
        return None
    url = os.getenv("UPSTASH_REDIS_URL")
    if not url:
        return None
    try:
        _redis_client = redis_sync.from_url(url, decode_responses=True)
    except Exception:
        _redis_client = None
    return _redis_client


def _local_get(key: str) -> Optional[Any]:
    if LOCAL_CACHE_TTL_SEC <= 0:
        return None
    hit = _LOCAL_CACHE.get(key)
    if not hit:
        return None
    expires_at, val = hit
    if time.time() < expires_at:
        return val
    _LOCAL_CACHE.pop(key, None)
    return None


def _local_set(key: str, val: Any) -> None:
    if LOCAL_CACHE_TTL_SEC <= 0:
        return
    _LOCAL_CACHE[key] = (time.time() + LOCAL_CACHE_TTL_SEC, val)


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    mem = _local_get(key)
    if mem is _MISS:
        return None
    if mem is not None:
        return mem

    client = _get_redis_client()
    if not client:
        return None

    try:
        cached = client.get(key)
        if cached and isinstance(cached, (str, bytes)):
            payload = json.loads(cached)
            _local_set(key, payload)
            return payload
        _local_set(key, _MISS)
        return None
    except Exception:
        return None


def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if not client:
        return
    try:
        client.set(key, json.dumps(payload), ex=SINGLE_STOCK_CACHE_TTL_SEC)
        _local_set(key, payload)
    except Exception:
        pass


def _build_cache_key(
    symbol: str,
    base_currency: str,
    needs_filings: bool,
    prefix: str = "openai:stock_report",
) -> str:
    asof_bucket = time.strftime("%Y-%m-%d", time.gmtime())
    return (
        f"{prefix}:{symbol.upper()}:{base_currency.upper()}:{int(needs_filings)}:"
        f"{asof_bucket}:{SINGLE_STOCK_SCHEMA_VERSION}"
    )


def _cap_timeout(requested: float, budget: TimeBudget) -> float:
    remaining = budget.remaining()
    if remaining <= 0:
        return 0.0
    return min(requested, remaining)


async def _timed_call(
    name: str,
    timeout_s: float,
    coro: Any,
    default: Any,
    data_gaps: list[str],
) -> Any:
    if timeout_s <= 0:
        data_gaps.append(f"{name} skipped due to global timeout budget")
        return default

    start = time.perf_counter()
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        data_gaps.append(f"{name} timed out after {timeout_s:.1f}s")
        return default
    except Exception as exc:
        data_gaps.append(f"{name} failed: {exc}")
        return default
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        print("[single_stock_analysis] %s duration_ms=%s", name, duration_ms)


def _build_citations(news_items: list[dict[str, Any]], filing_items: list[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    for item in news_items:
        citations.append(
            Citation.model_validate(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "published_at": item.get("published_at"),
                }
            )
        )
    for item in filing_items:
        citations.append(
            Citation.model_validate(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "published_at": item.get("published_at"),
                }
            )
        )
    return citations


def _build_fallback_report(
    *,
    symbol: str,
    as_of: str,
    data_gaps: list[str],
    citations: list[Citation],
    fundamentals: Optional[dict[str, Any]] = None,
) -> StockReport:
    normalized = (fundamentals or {}).get("normalized") if isinstance(fundamentals, dict) else {}
    snapshot = {
        "market_cap": normalized.get("market_cap"),
        "pe_ttm": normalized.get("pe_ttm"),
        "revenue_growth_yoy": normalized.get("revenue_growth_yoy"),
        "gross_margin": normalized.get("gross_margin"),
        "operating_margin": normalized.get("operating_margin"),
        "free_cash_flow": normalized.get("free_cash_flow"),
        "debt_to_equity": normalized.get("debt_to_equity"),
        "summary": "Fundamentals are limited or unavailable in this run.",
    }
    payload = {
        "symbol": symbol,
        "as_of": as_of,
        "quick_take": "Full synthesis unavailable due to data limits.",
        "what_changed_recently": [],
        "fundamentals_snapshot": snapshot,
        "catalysts_next_30_90d": [],
        "risks": [],
        "sentiment": {
            "overall": "neutral",
            "drivers": [],
            "sources": [],
        },
        "scenarios": {
            "bull": {
                "thesis": "Insufficient data to outline a bull case.",
                "key_assumptions": [],
                "watch_items": [],
                "sources": [],
            },
            "base": {
                "thesis": "Insufficient data to outline a base case.",
                "key_assumptions": [],
                "watch_items": [],
                "sources": [],
            },
            "bear": {
                "thesis": "Insufficient data to outline a bear case.",
                "key_assumptions": [],
                "watch_items": [],
                "sources": [],
            },
        },
        "confidence": {
            "score_0_100": 20,
            "rationale": "Report generated with limited data and/or timeouts.",
        },
        "citations": citations,
        "data_gaps": data_gaps,
    }
    return StockReport.model_validate(payload)


async def analyze_stock_async(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
    holdings: Optional[list[Any]] = None,
    *,
    allowed_domains: Optional[list[str]] = None,
    user_request: Optional[str] = None,
    needs_filings: bool = False,
    cik: Optional[str] = None,
) -> Dict[str, Any]:
    del holdings
    del allowed_domains

    pipeline_start = time.perf_counter()
    normalized_base_currency = (base_currency or "USD").upper()
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise ValueError("Missing symbol")

    needs_filings_flag = needs_filings_for_request(user_request, needs_filings)
    cache_key = _build_cache_key(
        symbol=clean_symbol,
        base_currency=normalized_base_currency,
        needs_filings=needs_filings_flag,
    )
    cached = _cache_get(cache_key)
    if cached:
        duration_ms = int((time.perf_counter() - pipeline_start) * 1000)
        print("[single_stock_analysis] cache_hit duration_ms=%s", duration_ms)
        return cached

    data_gaps: list[str] = []
    budget = TimeBudget(total_sec=GLOBAL_TIMEOUT_SEC)
    as_of = datetime.now(timezone.utc).isoformat()

    company_name = None
    if isinstance(metrics_for_symbol, dict):
        company_name = metrics_for_symbol.get("company_name") or metrics_for_symbol.get("name")

    tasks: list[tuple[str, Any]] = []
    news_timeout = _cap_timeout(NEWS_TIMEOUT_SEC, budget)
    if news_timeout > 0:
        tasks.append(
            (
                "news",
                _timed_call(
                    "news",
                    news_timeout,
                    fetch_news(clean_symbol, company_name=company_name),
                    [],
                    data_gaps,
                ),
            )
        )
    else:
        data_gaps.append("news skipped due to global timeout budget")

    fundamentals_timeout = _cap_timeout(FUNDAMENTALS_TIMEOUT_SEC, budget)
    if fundamentals_timeout > 0:
        tasks.append(
            (
                "fundamentals",
                _timed_call(
                    "fundamentals",
                    fundamentals_timeout,
                    fetch_fundamentals(clean_symbol, timeout_s=fundamentals_timeout),
                    FundamentalsResult({}, ["Fundamentals unavailable"]),
                    data_gaps,
                ),
            )
        )
    else:
        data_gaps.append("fundamentals skipped due to global timeout budget")

    if needs_filings_flag:
        filings_timeout = _cap_timeout(FILINGS_TIMEOUT_SEC, budget)
        if filings_timeout > 0:
            tasks.append(
                (
                    "filings",
                    _timed_call(
                        "filings",
                        filings_timeout,
                        fetch_filings(clean_symbol, needs_filings=True, cik=cik),
                        [],
                        data_gaps,
                    ),
                )
            )
        else:
            data_gaps.append("filings skipped due to global timeout budget")
    else:
        data_gaps.append("filings skipped (not requested)")

    results = await asyncio.gather(*[task for _, task in tasks])
    result_map = {name: result for (name, _), result in zip(tasks, results)}

    news_items = result_map.get("news") or []
    fundamentals_result = result_map.get("fundamentals")
    if not isinstance(fundamentals_result, FundamentalsResult):
        fundamentals_result = FundamentalsResult({}, ["Fundamentals unavailable"])
    filing_items = result_map.get("filings") or []

    data_gaps.extend(fundamentals_result.gaps)
    if not news_items:
        data_gaps.append("No recent news found")
    if needs_filings_flag and not filing_items:
        data_gaps.append("No recent filings found")

    citations = _build_citations(news_items, filing_items)
    inputs = {
        "symbol": clean_symbol,
        "as_of": as_of,
        "fundamentals": fundamentals_result.data,
        "news": news_items,
        "filings": filing_items,
        "data_gaps": data_gaps,
    }

    remaining_for_llm = _cap_timeout(LLM_TIMEOUT_SEC, budget)
    if remaining_for_llm <= 0:
        data_gaps.append("LLM synthesis skipped due to global timeout budget")
        report = _build_fallback_report(
            symbol=clean_symbol,
            as_of=as_of,
            data_gaps=data_gaps,
            citations=citations,
            fundamentals=fundamentals_result.data,
        )
        payload = report.model_dump()
        _cache_set(cache_key, payload)
        duration_ms = int((time.perf_counter() - pipeline_start) * 1000)
        print("[single_stock_analysis] completed_fallback duration_ms=%s", duration_ms)
        return payload

    try:
        llm_start = time.perf_counter()
        report = await synthesize_stock_report(
            inputs,
            model="gpt-5-mini",
            timeout_s=remaining_for_llm,
        )
        llm_ms = int((time.perf_counter() - llm_start) * 1000)
        print("[single_stock_analysis] llm duration_ms=%s", llm_ms)
    except Exception as exc:
        data_gaps.append(f"LLM synthesis failed: {exc}")
        report = _build_fallback_report(
            symbol=clean_symbol,
            as_of=as_of,
            data_gaps=data_gaps,
            citations=citations,
            fundamentals=fundamentals_result.data,
        )

    report = report.model_copy(
        update={
            "symbol": clean_symbol,
            "as_of": as_of,
            "data_gaps": data_gaps,
            "citations": citations,
        }
    )
    payload = report.model_dump()
    _cache_set(cache_key, payload)
    duration_ms = int((time.perf_counter() - pipeline_start) * 1000)
    print("[single_stock_analysis] completed duration_ms=%s", duration_ms)
    return payload
