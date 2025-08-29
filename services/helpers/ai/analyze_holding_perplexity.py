# services/ai_service_perplexity_basic.py
from __future__ import annotations

import os
import json
import re
import datetime as dt
from typing import Any, Dict

from openai import OpenAI  # Perplexity is OpenAI-compatible


# ----------------------------
# Config (env vars)
# ----------------------------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_BASE_URL = os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai")
PPLX_MODEL = os.getenv("PPLX_MODEL", "sonar-pro")  # or "sonar-reasoning-pro"

# Optional search controls (keep simple)
SEARCH_RECENCY = os.getenv("PPLX_SEARCH_RECENCY", "month")  # day|week|month|year
DOMAIN_FILTER = [d.strip() for d in os.getenv("PPLX_DOMAIN_FILTER", "").split(",") if d.strip()]

# ----------------------------
# Helpers
# ----------------------------
def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _to_dict(holding: Any) -> Dict[str, Any]:
    """Tolerant conversion of Pydantic v1/v2/dataclass/plain object -> dict."""
    if isinstance(holding, dict):
        return holding
    if hasattr(holding, "model_dump"):   # Pydantic v2
        return holding.model_dump()      # type: ignore[attr-defined]
    if hasattr(holding, "dict"):         # Pydantic v1
        return holding.dict()            # type: ignore[attr-defined]
    if hasattr(holding, "__dict__"):
        return dict(holding.__dict__)
    return {"symbol": str(holding)}

def _force_json(text: str) -> Dict[str, Any]:
    """Try to parse JSON; if not, try to extract the biggest {...} block; else wrap."""
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to grab a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback: wrap as rationale
    return {
        "rating": "watch",
        "rationale": text[:1200],
        "key_risks": [],
        "suggestions": [],
        "sources": [],
        "disclaimer": "This is educational information, not financial advice.",
    }

def _normalize_rating(v: Any) -> str:
    r = str(v or "watch").lower().strip()
    return r if r in {"hold", "sell", "watch", "diversify"} else "watch"


# ----------------------------
# Public function
# ----------------------------
def analyze_investment_holding_pplx(holding: Any) -> Dict[str, Any]:
    """
    Fresh Perplexity-only analysis (no Yahoo/news you provide).
    Returns a dict your UI can display:
      rating, rationale, key_risks[], suggestions[], data_notes[], disclaimer, sources[].
    """
    if not PPLX_API_KEY:
        return {
            "error": "missing_perplexity_api_key",
            "message": "Set PPLX_API_KEY in your environment.",
        }

    h = _to_dict(holding)
    symbol = (h.get("symbol") or "").upper().strip()
    name = (h.get("name") or "").strip()
    asset_type = (h.get("type") or "stock").lower().strip()

    if not symbol:
        return {"error": "missing_symbol"}

    client = OpenAI(api_key=PPLX_API_KEY, base_url=PPLX_BASE_URL)

    # Keep the prompt minimal but clear; ask for strict JSON keys.
    system = (
    "You are a cautious retail investing explainer. Use web search to gather the latest, "
    "reliable facts (dividends, earnings, analyst actions, material risks). "
    "Prefer primary/company sources (IR pages, SEC filings, official press releases) and major outlets "
    "(Reuters, AP, WSJ, Barron's). Avoid YouTube, opinion blogs, and technical analysis signals. "
    "Do not provide personalized financial advice or price targets. "
    "Return VALID JSON ONLY with keys: "
    "rating (hold|sell|watch|diversify), rationale (string), key_risks (string[]), "
    "suggestions (string[]), sources ({title?, url}[]), events ({date?, label, url?}[]), "
    "next_dates ({earnings_date?: string, ex_dividend_date?: string}), disclaimer (string)."
)

    user_msg = {
    "holding": {"symbol": symbol, "name": name, "type": asset_type},
    "task": (
        "Analyze this security for a retail investor.\n"
        "- Focus on **what changed in the last 30 days** (earnings, dividend, guidance, rating actions, litigation).\n"
        "- Provide a conservative single-word rating from {hold, sell, watch, diversify} with a concise rationale.\n"
        "- Give 3–5 risks and 2–4 suggestions (≤12 words each).\n"
        "- Extract **upcoming dates**: next earnings_date and ex_dividend_date if published by trusted sources.\n"
        "- Add 2–6 reputable sources (no YouTube, no TA blogs).\n"
        "Return ONLY JSON. No extra text."
    ),
    "json_example": {
        "rating": "watch",
        "rationale": "Stable NIM; trimmed loan growth; dividend maintained; watch CRE exposure.",
        "key_risks": ["CRE stress rising", "Regional economy sensitivity", "Funding cost pressure"],
        "suggestions": ["Monitor next earnings call", "Diversify across regions", "Track payout ratio"],
        "sources": [{"title": "Q2 press release", "url": "https://..."}],
        "events": [{"date": "2025-08-21", "label": "Ex-dividend", "url": "https://..."}],
        "next_dates": {"earnings_date": "2025-10-24", "ex_dividend_date": "2025-11-21"},
        "disclaimer": "This is educational information, not financial advice."
    },
}

    # Perplexity search knobs (optional but helpful)
    extra = {
        "search_recency_filter": SEARCH_RECENCY,  # day|week|month|year
        "return_images": False,
    }
    if DOMAIN_FILTER:
        extra["search_domain_filter"] = DOMAIN_FILTER

    # Simple call; no response_format (keeps things SDK-agnostic)
    resp = client.chat.completions.create(
        model=PPLX_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0.2,
        max_tokens=900,
        extra_body=extra,
    )

    raw_txt = (resp.choices[0].message.content or "{}").strip()
    obj = _force_json(raw_txt)

    rating = _normalize_rating(obj.get("rating"))
    analysis = {
        "symbol": symbol,
        "as_of_utc": _now_utc_iso(),
        "pnl_abs": None,          # unknown (we're not doing portfolio math here)
        "pnl_pct": None,          # unknown
        "market_context": {},     # intentionally empty (no Yahoo)
        "rating": rating,
        "rationale": obj.get("rationale", ""),
        "key_risks": obj.get("key_risks", []) or [],
        "suggestions": obj.get("suggestions", []) or [],
        "data_notes": [
            "Live analysis via Perplexity Sonar (web search).",
            f"Model: {PPLX_MODEL}, Recency: {SEARCH_RECENCY}",
        ],
        "disclaimer": obj.get("disclaimer", "This is educational information, not financial advice."),
        # optional for UI:
        "sources": obj.get("sources", []) or [],
        "provider_used": "perplexity",
    }
    return analysis
