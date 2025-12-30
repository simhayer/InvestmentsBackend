# services/linkup/agents/single_stock_agent.py

from __future__ import annotations

import hashlib
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
    WebSearchTool,
)
from agents.models.default_models import get_default_model_settings
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate
from openai.types.responses.web_search_tool import Filters as WebSearchToolFilters
from services.linkup.metrics.stock_metrics import StockMetricsCalculator
from services.linkup.schemas.single_stock_analysis_schema import SINGLE_STOCK_ANALYSIS_SCHEMA
from services.linkup.linkup_search import linkup_structured_search
from utils.common_helpers import unwrap_linkup
try:
    import redis as redis_sync  # redis-py for direct Redis/Upstash (redis:// / rediss://)
except Exception:
    redis_sync = None

SINGLE_STOCK_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_CACHE_TTL_SEC", "86400"))  # default 24h
LOCAL_CACHE_TTL_SEC = int(os.getenv("SINGLE_STOCK_LOCAL_CACHE_TTL_SEC", "60"))  # reduce Redis reads

_redis_client = None

def _get_redis_client():
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

# small in-process cache to avoid repeated Redis GETs on the same key within a short window
_LOCAL_CACHE: dict[str, tuple[float, Any]] = {}
_MISS = object()

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

# ============================================================
# 1. OpenAI Structured Output (single stock analysis)
# ============================================================

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
- Whenever you rely on an external source, include inline citations in the form 【source†L#-L#】 inside the relevant text field.
- If you cannot find solid support for a detail, speak qualitatively or omit it. Do NOT guess.

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

NEWS_PROMPT = """
You are a financial news researcher.
Find recent material news and events for the specified company (last 30-90 days where possible).
Use web search and cite sources inline using 【source†L#-L#】.
Focus on earnings, guidance, product launches, regulatory actions, major deals, and leadership changes.
Avoid investment recommendations or personalized advice.
Return plain text only.
""".strip()

FILINGS_PROMPT = """
You are a filings and investor materials analyst.
Use recent filings, investor presentations, and reputable sources to summarize:
- Business model and revenue drivers
- Qualitative financial/operating trends (no fabricated numbers)
- Competitive positioning and notable structural tailwinds/headwinds
Include inline citations as 【source†L#-L#】.
Avoid investment recommendations or personalized advice.
Return plain text only.
""".strip()


def _get_openai_model() -> str:
    # Set OPENAI_MODEL to gpt-5.2-mini if available in your org; default is gpt-5-mini.
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


def _get_allowed_domains(allowed_domains: Optional[list[str]]) -> Optional[list[str]]:
    if allowed_domains:
        return allowed_domains
    raw = os.getenv("OPENAI_WEB_ALLOWED_DOMAINS", "").strip()
    if not raw:
        return None
    domains = [domain.strip() for domain in raw.split(",") if domain.strip()]
    return domains or None


def _apply_strict_required(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structured Outputs with strict=True require 'required' to list all properties
    for object schemas. Ensure that across the schema tree.
    """
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


class SingleStockAnalysisOutputSchema(AgentOutputSchemaBase):
    def __init__(self, schema: Dict[str, Any]) -> None:
        self._schema = schema
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


def _build_model_settings(model: str) -> ModelSettings:
    base_settings = get_default_model_settings(model)
    return base_settings.resolve(
        ModelSettings(
            tool_choice="auto",
            response_include=["web_search_call.action.sources"],
        )
    )


def _build_web_search_tool(allowed_domains: Optional[list[str]]) -> WebSearchTool:
    filters = WebSearchToolFilters(allowed_domains=allowed_domains) if allowed_domains else None
    return WebSearchTool(filters=filters)


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


def _run_text_agent(agent: Agent[Any], input_text: str) -> str:
    try:
        result = Runner.run_sync(agent, input=input_text, max_turns=2)
        output = result.final_output
        return output if isinstance(output, str) else json.dumps(output, default=str)
    except Exception:
        return ""


def _repair_json(
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
    repair_result = Runner.run_sync(repair_agent, input=repair_input, max_turns=1)
    repaired = repair_result.final_output
    if not isinstance(repaired, dict):
        raise ModelBehaviorError("Repair output was not a JSON object.")
    return repaired


def analyze_stock(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
    holdings: Optional[list[Any]] = None,
    *,
    allowed_domains: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Returns a dict that conforms to SINGLE_STOCK_ANALYSIS_SCHEMA.
    """
    normalized_base_currency = (base_currency or "USD").upper()
    metrics_payload = dict(metrics_for_symbol or {})
    if holdings:
        calculator = StockMetricsCalculator(base_currency=normalized_base_currency)
        computed = calculator.build_for_symbol_sync(symbol, holdings)
        if computed and "computed_metrics" not in metrics_payload:
            metrics_payload["computed_metrics"] = computed

    cache_key = _build_cache_key(
        symbol=symbol,
        base_currency=normalized_base_currency,
        metrics_for_symbol=metrics_payload,
        prefix="openai:single_stock",
    )

    cached = _cache_get(cache_key)
    if cached:
        return cached

    model = _get_openai_model()
    model_settings = _build_model_settings(model)

    domains = _get_allowed_domains(allowed_domains)
    web_tool = _build_web_search_tool(domains)
    tools = [web_tool]

    user_payload = {
        "symbol": symbol,
        "base_currency": normalized_base_currency,
        "metrics_for_symbol": metrics_payload,
    }

    news_agent = Agent(
        name="Stock news researcher",
        instructions=NEWS_PROMPT,
        model=model,
        model_settings=model_settings,
        tools=tools,
    )
    filings_agent = Agent(
        name="Stock filings researcher",
        instructions=FILINGS_PROMPT,
        model=model,
        model_settings=model_settings,
        tools=tools,
    )
    news_context = _run_text_agent(
        news_agent,
        f"Find recent material news and events for {symbol}.",
    )
    filings_context = _run_text_agent(
        filings_agent,
        f"Summarize recent filings and investor materials for {symbol}.",
    )

    output_schema = SingleStockAnalysisOutputSchema(SINGLE_STOCK_ANALYSIS_SCHEMA)
    analysis_agent = Agent(
        name="Single stock analyst",
        instructions=SYSTEM_PROMPT,
        model=model,
        model_settings=model_settings,
        tools=tools,
        output_type=output_schema,
    )
    analysis_input = (
        "Analyze the stock using the provided schema and constraints.\n"
        f"Stock context (JSON): {json.dumps(user_payload, default=str)}\n"
        f"News context (text): {news_context}\n"
        f"Filings context (text): {filings_context}\n"
        "Return ONLY JSON that matches the schema."
    )

    try:
        result = Runner.run_sync(analysis_agent, input=analysis_input, max_turns=3)
        data = result.final_output
        if not isinstance(data, dict):
            raise ModelBehaviorError("Analysis output was not a JSON object.")
    except ModelBehaviorError as exc:
        raw = _extract_raw_output_text(exc)
        if not raw:
            raise
        data = _repair_json(
            raw_json=raw,
            model=model,
            model_settings=model_settings,
            output_schema=output_schema,
        )

    try:
        jsonschema_validate(instance=data, schema=SINGLE_STOCK_ANALYSIS_SCHEMA)
    except ValidationError:
        raw = json.dumps(data, ensure_ascii=False)
        data = _repair_json(
            raw_json=raw,
            model=model,
            model_settings=model_settings,
            output_schema=output_schema,
        )
        jsonschema_validate(instance=data, schema=SINGLE_STOCK_ANALYSIS_SCHEMA)

    _cache_set(cache_key, data)
    return data


# ============================================================
# 3. Cache helpers (shared across users)
# ============================================================

def _build_cache_key(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]],
    prefix: str = "linkup:single_stock",
) -> str:
    """
    Build a deterministic key so cached analyses are reused across users.
    Metrics are hashed to keep the key small.
    """
    metrics_hash = "nometrics"
    if metrics_for_symbol:
        try:
            blob = json.dumps(metrics_for_symbol, sort_keys=True, default=str)
        except Exception:
            blob = repr(metrics_for_symbol)
        metrics_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    return f"{prefix}:{symbol.upper()}:{base_currency.upper()}:{metrics_hash}"


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
        # Fail soft if Redis is unavailable
        pass

