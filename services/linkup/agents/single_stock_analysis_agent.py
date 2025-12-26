# services/linkup/agents/single_stock_agent.py

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
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
# 2. Instruction builder for Linkup / LLM
# ============================================================

def build_single_stock_query(
    symbol: str,
    base_currency: str = "USD",
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return {
        "role": (
            "You are a professional equity research analyst. "
            "You produce a structured, in-depth but accessible profile for a single stock. "
            "You MUST return ONLY valid JSON that matches SINGLE_STOCK_SUMMARY_SCHEMA. "
            "You are not permitted to give personalized investment advice or trading recommendations."
        ),
        "step_1_task": [
            "Explain what this company does, how it makes money, and where it sits in its industry.",
            "Summarize financial and operating trends (revenue, profitability, leverage, cash flows) "
            "based on recent filings and credible sources, without fabricating exact numbers.",
            "Describe recent material events (earnings, guidance changes, major product launches, "
            "regulatory actions, large deals) and their likely relevance for investors.",
            "Summarize current sentiment and typical investment thesis (bull, base, bear).",
            "Outline key risks, structural tailwinds/headwinds, and provide a qualitative valuation context.",
        ],
        "step_2_context": [
            f"Focus on the stock identified by ticker symbol: {symbol}.",
            f"Assume the user's base currency is {base_currency}. Currency matters only for narrative context; "
            "do not compute portfolio-level P&L here.",
            "Treat deterministic metrics in metrics_for_symbol as read-only facts if provided. "
            "You may reference them qualitatively (e.g. 'The stock has been volatile recently'), "
            "but do NOT override or recompute them.",
            "Assume a sophisticated but non-professional audience: avoid jargon where possible, "
            "explain it briefly when unavoidable.",
        ],
        "step_3_references": [
            "Use credible sources: company filings, investor presentations, major financial news outlets, "
            "and reliable market data providers.",
            "Whenever you rely on an external source, include inline citations in the form 【source†L#-L#】 "
            "inside the relevant text field.",
            "If you cannot find solid support for a specific detail (e.g., an exact margin or revenue number), "
            "either speak qualitatively (e.g. 'high', 'moderate', 'declining') or omit the detail. Do NOT guess.",
        ],
        "step_4_evaluate": [
            "Check that your narrative is internally consistent and grounded in cited sources.",
            "Avoid unsupported strong claims such as 'this stock is cheap/expensive' unless backed by clear context; "
            "prefer 'often discussed as richly valued vs peers' with citations.",
            "Ensure you do NOT use words that can be interpreted as direct investment advice: "
            "avoid 'you should buy', 'you should sell', 'this is a buy/hold/sell', 'add', 'trim', or "
            "specific allocation instructions.",
            "If you are uncertain or the data is limited (e.g., for a very small or illiquid name), "
            "lower the relevant section_confidence and mention these limitations explicitly.",
        ],
        "step_5_iterate": [
            "After drafting the full JSON, perform one pass to simplify wording where possible, "
            "and to remove any implied recommendation or prescriptive language.",
            "Make sure the disclaimer clearly states that this is general, informational analysis only, "
            "not personalized advice.",
        ],
        "constraints": [
            "STRICTLY conform to SINGLE_STOCK_SUMMARY_SCHEMA.",
            "Return ONLY JSON, no markdown, no prose outside the JSON.",
            "Do NOT make up detailed numeric fundamentals (revenue, EPS, margins, price targets) "
            "if they are not clearly supported by sources.",
            "It is always acceptable to say 'information not clearly available from public sources' "
            "rather than hallucinate.",
            "All content is informational and educational, not investment, tax, or legal advice.",
        ],
    "stock_context": {
            "symbol": symbol,
            "base_currency": base_currency,
            "metrics_for_symbol": metrics_for_symbol or {},
        },
    }


# ============================================================
# 3. Cache helpers (shared across users)
# ============================================================

def _build_cache_key(
    symbol: str,
    base_currency: str,
    metrics_for_symbol: Optional[Dict[str, Any]],
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

    return f"linkup:single_stock:{symbol.upper()}:{base_currency.upper()}:{metrics_hash}"


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


# ============================================================
# 4. Convenience function to call Linkup
# ============================================================

def call_link_up_for_single_stock(
    symbol: str,
    base_currency: Optional[str] = None,
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_base_currency = (base_currency or "USD").upper()
    cache_key = _build_cache_key(
        symbol=symbol,
        base_currency=normalized_base_currency,
        metrics_for_symbol=metrics_for_symbol,
    )

    cached = _cache_get(cache_key)
    if cached:
        return cached

    instruction = build_single_stock_query(
        symbol=symbol,
        base_currency=normalized_base_currency,
        metrics_for_symbol=metrics_for_symbol,
    )

    try:    
        response = unwrap_linkup(linkup_structured_search(
            query_obj=instruction,
            schema=SINGLE_STOCK_ANALYSIS_SCHEMA,
            days=30,
            include_sources=False,
            depth="standard",
            max_retries=2,
        )
    )
    except Exception as e:
        # logger.error(f"Error occurred while fetching portfolio AI layers: {e}")
        return {"error": str(e)}

    # Assuming Linkup client already returns Python dict. If it's a string, json.loads it here.
    if isinstance(response, dict) and not response.get("error"):
        _cache_set(cache_key, response)
    return response
