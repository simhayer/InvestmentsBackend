"""OpenAI Agents-based single stock analysis pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from agents import (
    Agent,
    AgentOutputSchemaBase,
    ItemHelpers,
    ModelBehaviorError,
    ModelSettings,
    Runner,
)
from agents.models.default_models import get_default_model_settings
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate
from services.linkup.linkup_search import linkup_structured_search
from services.linkup.metrics.stock_metrics import StockMetricsCalculator
from services.linkup.schemas.single_stock_analysis_schema import SINGLE_STOCK_ANALYSIS_SCHEMA

try:
    import redis as redis_sync  # redis-py for direct Redis/Upstash (redis:// / rediss://)
except Exception:
    redis_sync = None

SINGLE_STOCK_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_CACHE_TTL_SEC", "86400"))
LOCAL_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_LOCAL_CACHE_TTL_SEC", "60"))
SINGLE_STOCK_SCHEMA_VERSION = os.getenv("SINGLE_STOCK_SCHEMA_VERSION", "2")
NEWS_LOOKBACK_DAYS = int(os.getenv("SINGLE_STOCK_NEWS_LOOKBACK_DAYS", "60"))
FILINGS_LOOKBACK_DAYS = int(os.getenv("SINGLE_STOCK_FILINGS_LOOKBACK_DAYS", "365"))
NEWS_MAX_ITEMS = int(os.getenv("SINGLE_STOCK_NEWS_MAX_ITEMS", "5"))
FILINGS_MAX_ITEMS = int(os.getenv("SINGLE_STOCK_FILINGS_MAX_ITEMS", "4"))
NEWS_CONTEXT_MAX_CHARS = int(os.getenv("SINGLE_STOCK_NEWS_CONTEXT_MAX_CHARS", "2500"))
FILINGS_CONTEXT_MAX_CHARS = int(os.getenv("SINGLE_STOCK_FILINGS_CONTEXT_MAX_CHARS", "4500"))
NEWS_ITEM_HEADLINE_MAX_CHARS = int(os.getenv("SINGLE_STOCK_NEWS_HEADLINE_MAX_CHARS", "160"))
NEWS_ITEM_SUMMARY_MAX_CHARS = int(os.getenv("SINGLE_STOCK_NEWS_SUMMARY_MAX_CHARS", "380"))
FILINGS_ITEM_POINT_MAX_CHARS = int(os.getenv("SINGLE_STOCK_FILINGS_POINT_MAX_CHARS", "240"))
FILINGS_MAX_POINTS_PER_ITEM = int(os.getenv("SINGLE_STOCK_FILINGS_MAX_POINTS", "4"))

SYSTEM_PROMPT = """
You are a professional equity research analyst.
You produce a structured, in-depth but accessible profile for a single stock.
You MUST return ONLY valid JSON that matches the provided JSON schema.
You are not permitted to give personalized investment advice or trading recommendations.

Task:
- Explain what the company does, how it makes money, and where it sits in its industry.
- Summarize financial and operating trends qualitatively based on recent filings and credible sources, without fabricating exact numbers.
- Describe recent material events (earnings, guidance changes, major product launches, regulatory actions, large deals) and likely relevance for investors.
- Summarize current sentiment and typical investment thesis (bull, base, bear).
- Outline key risks, structural tailwinds/headwinds, and provide qualitative valuation context.

References:
- Use credible sources (filings, investor materials, major financial news outlets, reliable market data).
- Whenever you rely on an external source, include inline citations using Linkup source IDs like 【linkup:news_1】 or 【linkup:filing_2】 from SOURCES_MAP.
- If you cannot find solid support for a detail, speak qualitatively or omit it. Do NOT guess.
- Populate recent_developments from NEWS_SUMMARIES and copy source/url from SOURCES_MAP when possible.

Constraints:
- STRICTLY conform to the provided JSON schema.
- Return ONLY JSON (no markdown).
- Do NOT use direct recommendation language like: "you should buy/sell/hold", "add", "trim".
- If information is limited, explicitly say so in limitations and lower confidence scores.
- If information is insufficient, return empty arrays/objects and explain the limitation.

Output quality:
- Prefer simple explanations; define jargon briefly if used.
- If uncertain, reflect that in section_confidence and confidence_overall.
""".strip()

LINKUP_NEWS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string"},
                    "publisher": {"type": "string"},
                    "url": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["headline", "date", "publisher", "url", "summary"],
            },
        }
    },
    "required": ["items"],
}

LINKUP_FILINGS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "filing_type": {"type": "string"},
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["filing_type", "date", "source", "url", "key_points"],
            },
        }
    },
    "required": ["items"],
}

_redis_client: Optional[Any] = None
_LOCAL_CACHE: dict[str, tuple[float, Any]] = {}
_MISS = object()


def _get_openai_model() -> str:
    """Selects the default OpenAI model for this pipeline."""
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


def _apply_strict_required(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure strict JSON schema requirements for all nested objects."""

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                props = node["properties"]
                node["required"] = list(props.keys())
                for val in props.values():
                    _walk(val)
            elif node.get("type") == "array" and isinstance(node.get("items"), dict):
                _walk(node["items"])
            else:
                for val in node.values():
                    _walk(val)
        elif isinstance(node, list):
            for val in node:
                _walk(val)

    schema_copy = deepcopy(schema)
    _walk(schema_copy)
    return schema_copy


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


def _build_cache_key(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]],
    prefix: str = "openai:single_stock",
) -> str:
    """Build a deterministic cache key so analyses are reused across users."""
    del metrics_for_symbol
    asof_bucket = time.strftime("%Y-%m-%d", time.gmtime())
    return (
        f"{prefix}:{symbol.upper()}:{base_currency.upper()}:"
        f"{asof_bucket}:{SINGLE_STOCK_SCHEMA_VERSION}"
    )


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


class SingleStockAnalysisOutputSchema(AgentOutputSchemaBase):
    def __init__(self, schema: Dict[str, Any]) -> None:
        self._strict_schema = _apply_strict_required(schema)

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "single_stock_analysis"

    def json_schema(self) -> Dict[str, Any]:
        return self._strict_schema

    def is_strict_json_schema(self) -> bool:
        return True

    def validate_json(self, json_str: str) -> Any:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ModelBehaviorError(f"Invalid JSON: {exc}") from exc


def _build_analysis_model_settings(model: str) -> ModelSettings:
    base_settings = get_default_model_settings(model)
    return base_settings.resolve(ModelSettings(tool_choice="none"))


def _extract_raw_output_text(exc: ModelBehaviorError) -> Optional[str]:
    run_data = getattr(exc, "run_data", None)
    if not run_data or not run_data.raw_responses:
        return None
    last_response = run_data.raw_responses[-1]
    for item in reversed(last_response.output):
        text = ItemHelpers.extract_last_text(item)
        if text:
            return text
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _coerce_linkup_data(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if hasattr(payload, "data") and not isinstance(payload, dict):
        return _coerce_linkup_data(getattr(payload, "data"))
    if hasattr(payload, "model_dump"):
        try:
            payload = payload.model_dump()
        except Exception:
            pass
    if isinstance(payload, dict):
        if "data" in payload and "sources" in payload and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload
    return None


def _extract_linkup_items(response: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not response or not response.get("ok"):
        return []
    data = _coerce_linkup_data(response.get("data"))
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _build_linkup_news_query(symbol: str) -> Dict[str, Any]:
    return {
        "role": "You are a financial news researcher using Linkup as deterministic retrieval.",
        "task": [
            f"Find the 3-5 most material news items about {symbol} from the last {NEWS_LOOKBACK_DAYS} days.",
            "Prefer earnings, guidance changes, product launches, regulatory actions, major deals, and leadership changes.",
            "If there is limited material news, return fewer items rather than guessing.",
        ],
        "output": [
            "Return JSON that matches the provided schema.",
            "Each summary must be 1-2 concise sentences with no invented numbers or dates.",
        ],
        "constraints": [
            "Use only credible sources returned by Linkup.",
            "Do not fabricate dates, events, or metrics.",
            "Avoid investment recommendations.",
        ],
    }


def _build_linkup_filings_query(symbol: str) -> Dict[str, Any]:
    return {
        "role": "You are a filings and investor materials analyst using Linkup as deterministic retrieval.",
        "task": [
            f"Find the latest annual and quarterly filings for {symbol} from the last {FILINGS_LOOKBACK_DAYS} days.",
            "Include major investor-relations updates (earnings releases or investor presentations) if available.",
            "Summaries must be bullet-style key points, not full document summaries.",
        ],
        "output": [
            "Return JSON that matches the provided schema.",
            "key_points must be 2-4 short bullets per item.",
        ],
        "constraints": [
            "Use only official filings or company investor-relations sources returned by Linkup.",
            "Do not fabricate dates, document types, or metrics.",
            "Avoid investment recommendations.",
        ],
    }


def _prepare_news_items(raw_items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cleaned: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        headline = _normalize_text(item.get("headline") or item.get("title"))
        url = _normalize_text(item.get("url") or item.get("link"))
        if not headline or not url:
            continue
        summary = _normalize_text(item.get("summary") or "")
        if not summary:
            summary = headline
        date = _normalize_text(item.get("date") or "")
        source = _normalize_text(item.get("publisher") or item.get("source") or "")
        headline = _truncate_text(headline, NEWS_ITEM_HEADLINE_MAX_CHARS)
        summary = _truncate_text(summary, NEWS_ITEM_SUMMARY_MAX_CHARS)
        dedupe_key = url or headline.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(
            {
                "headline": headline,
                "date": date or "unknown",
                "source": source or "unknown",
                "url": url,
                "summary": summary,
            }
        )
        if len(cleaned) >= NEWS_MAX_ITEMS:
            break
    while cleaned and len(_format_news_summaries(cleaned)) > NEWS_CONTEXT_MAX_CHARS:
        cleaned.pop()
    return cleaned


def _prepare_filing_items(raw_items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cleaned: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        filing_type = _normalize_text(item.get("filing_type") or item.get("type") or "")
        title = _normalize_text(item.get("title") or "")
        url = _normalize_text(item.get("url") or item.get("link"))
        if not url or not (filing_type or title):
            continue
        date = _normalize_text(item.get("date") or "")
        source = _normalize_text(item.get("source") or item.get("publisher") or "")
        key_points = item.get("key_points") or item.get("bullet_points")
        points: list[str] = []
        if isinstance(key_points, list):
            for point in key_points:
                text = _normalize_text(point)
                if text:
                    points.append(_truncate_text(text, FILINGS_ITEM_POINT_MAX_CHARS))
        elif isinstance(key_points, str):
            text = _normalize_text(key_points)
            if text:
                points.append(_truncate_text(text, FILINGS_ITEM_POINT_MAX_CHARS))
        if not points:
            summary = _normalize_text(item.get("summary") or "")
            if summary:
                points = [_truncate_text(summary, FILINGS_ITEM_POINT_MAX_CHARS)]
        points = points[:FILINGS_MAX_POINTS_PER_ITEM]
        dedupe_key = url or f"{filing_type}:{title}".lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(
            {
                "filing_type": filing_type or "Filing",
                "title": title,
                "date": date or "unknown",
                "source": source or "unknown",
                "url": url,
                "key_points": points,
            }
        )
        if len(cleaned) >= FILINGS_MAX_ITEMS:
            break
    while cleaned and len(_format_filings_summaries(cleaned)) > FILINGS_CONTEXT_MAX_CHARS:
        cleaned.pop()
    return cleaned


def _format_sources_map(
    news_items: list[Dict[str, Any]],
    filing_items: list[Dict[str, Any]],
) -> str:
    lines = ["SOURCES_MAP:"]
    for idx, item in enumerate(news_items, start=1):
        lines.append(
            f"[linkup:news_{idx}] {item.get('source','unknown')} | {item.get('date','unknown')} | {item.get('url','')}"
        )
    for idx, item in enumerate(filing_items, start=1):
        lines.append(
            f"[linkup:filing_{idx}] {item.get('source','unknown')} | {item.get('date','unknown')} | {item.get('url','')}"
        )
    if len(lines) == 1:
        lines.append("(none)")
    return "\n".join(lines)


def _format_news_summaries(news_items: list[Dict[str, Any]]) -> str:
    lines = ["NEWS_SUMMARIES:"]
    if not news_items:
        lines.append("(none)")
        return "\n".join(lines)
    for idx, item in enumerate(news_items, start=1):
        lines.append(
            "- (linkup:news_{idx}) {date} | {source} | Headline: {headline}. Summary: {summary}".format(
                idx=idx,
                date=item.get("date", "unknown"),
                source=item.get("source", "unknown"),
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
            )
        )
    return "\n".join(lines)


def _format_filings_summaries(filing_items: list[Dict[str, Any]]) -> str:
    lines = ["FILINGS_SUMMARIES:"]
    if not filing_items:
        lines.append("(none)")
        return "\n".join(lines)
    for idx, item in enumerate(filing_items, start=1):
        points = "; ".join(item.get("key_points") or [])
        if not points:
            points = "Summary unavailable."
        title = item.get("title")
        filing_label = item.get("filing_type", "Filing")
        if title and title.lower() not in filing_label.lower():
            label = f"{filing_label} | {title}"
        else:
            label = filing_label
        lines.append(
            "- (linkup:filing_{idx}) {date} | {source} | {label} | Key points: {points}".format(
                idx=idx,
                date=item.get("date", "unknown"),
                source=item.get("source", "unknown"),
                label=label,
                points=points,
            )
        )
    return "\n".join(lines)


async def fetch_news_via_linkup(symbol: str) -> list[Dict[str, Any]]:
    response = await asyncio.to_thread(
        linkup_structured_search,
        query_obj=_build_linkup_news_query(symbol),
        schema=LINKUP_NEWS_SCHEMA,
        days=NEWS_LOOKBACK_DAYS,
        include_sources=False,
        depth="standard",
        max_retries=2,
    )
    return _prepare_news_items(_extract_linkup_items(response))


async def fetch_filings_via_linkup(symbol: str) -> list[Dict[str, Any]]:
    response = await asyncio.to_thread(
        linkup_structured_search,
        query_obj=_build_linkup_filings_query(symbol),
        schema=LINKUP_FILINGS_SCHEMA,
        days=FILINGS_LOOKBACK_DAYS,
        include_sources=False,
        depth="standard",
        max_retries=2,
    )
    return _prepare_filing_items(_extract_linkup_items(response))


async def _repair_json_async(
    *,
    raw_json: str,
    model: str,
    model_settings: ModelSettings,
    output_schema: AgentOutputSchemaBase,
) -> Dict[str, Any]:
    repair_agent = Agent(
        name="Single stock JSON repair",
        instructions=(
            "You fix JSON to match a schema. Output ONLY valid JSON. "
            "Do not add new analysis or facts."
        ),
        model=model,
        model_settings=model_settings,
        output_type=output_schema,
    )
    repair_input = (
        "Fix this JSON to match the schema strictly. Do not add new analysis.\n"
        f"JSON:\n{raw_json}"
    )
    repair_result = await Runner.run(repair_agent, input=repair_input, max_turns=1)
    repaired = repair_result.final_output
    if not isinstance(repaired, dict):
        raise ModelBehaviorError("Repair output was not a JSON object.")
    return repaired


async def analyze_stock_async(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
    holdings: Optional[list[Any]] = None,
    *,
    allowed_domains: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Return a dict that conforms to SINGLE_STOCK_ANALYSIS_SCHEMA."""
    normalized_base_currency = (base_currency or "USD").upper()
    metrics_payload = dict(metrics_for_symbol or {})

    if holdings:
        calculator = StockMetricsCalculator(base_currency=normalized_base_currency)
        computed = await calculator.build_for_symbol(symbol, holdings)
        if computed and "computed_metrics" not in metrics_payload:
            metrics_payload["computed_metrics"] = computed

    cache_key = _build_cache_key(
        symbol=symbol,
        base_currency=normalized_base_currency,
        metrics_for_symbol=metrics_payload,
    )
    cached = _cache_get(cache_key)
    if cached:
        return cached

    model = _get_openai_model()
    analysis_model_settings = _build_analysis_model_settings(model)

    user_payload = {
        "symbol": symbol,
        "base_currency": normalized_base_currency,
        "metrics_for_symbol": metrics_payload,
    }

    del allowed_domains
    news_items, filing_items = await asyncio.gather(
        fetch_news_via_linkup(symbol),
        fetch_filings_via_linkup(symbol),
    )
    sources_map = _format_sources_map(news_items, filing_items)
    news_context = _format_news_summaries(news_items)
    filings_context = _format_filings_summaries(filing_items)

    output_schema = SingleStockAnalysisOutputSchema(SINGLE_STOCK_ANALYSIS_SCHEMA)
    analysis_agent = Agent(
        name="Single stock analyst",
        instructions=SYSTEM_PROMPT,
        model=model,
        model_settings=analysis_model_settings,
        tools=[],
        output_type=output_schema,
    )
    analysis_input = (
        "Analyze the stock using the provided schema and constraints.\n"
        f"Stock context (JSON): {json.dumps(user_payload, default=str)}\n"
        f"{sources_map}\n"
        f"{news_context}\n"
        f"{filings_context}\n"
        "Use NEWS_SUMMARIES to populate recent_developments (3-5 items if available). "
        "Copy headline/date/source/url from the matching Linkup items.\n"
        "Cite sources using Linkup IDs from SOURCES_MAP (e.g., 【linkup:news_1】).\n"
        "Return ONLY JSON that matches the schema."
    )

    try:
        result = await Runner.run(analysis_agent, input=analysis_input, max_turns=2)
        data = result.final_output
        if not isinstance(data, dict):
            raise ModelBehaviorError("Analysis output was not a JSON object.")
    except ModelBehaviorError as exc:
        raw = _extract_raw_output_text(exc)
        if not raw:
            raise
        print("Repairing invalid JSON from analysis agent...")
        data = await _repair_json_async(
            raw_json=raw,
            model=model,
            model_settings=analysis_model_settings,
            output_schema=output_schema,
        )

    try:
        jsonschema_validate(instance=data, schema=SINGLE_STOCK_ANALYSIS_SCHEMA)
    except ValidationError:
        print("Final JSON validation failed, repairing...")
        raw = json.dumps(data, ensure_ascii=False)
        data = await _repair_json_async(
            raw_json=raw,
            model=model,
            model_settings=analysis_model_settings,
            output_schema=output_schema,
        )
        jsonschema_validate(instance=data, schema=SINGLE_STOCK_ANALYSIS_SCHEMA)

    _cache_set(cache_key, data)
    return data
