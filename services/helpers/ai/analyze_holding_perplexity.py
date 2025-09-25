import os
from typing import Any, Dict, List, Optional, Tuple
import json
from openai import OpenAI
from services.helpers.ai.json_helpers import S, E, extract_json

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_BASE_URL = os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai")
PPLX_MODEL = os.getenv("PPLX_MODEL", "sonar-pro")  # e.g., sonar-pro / sonar-reasoning-pro
SEARCH_RECENCY = os.getenv("PPLX_SEARCH_RECENCY", "month")  # day|week|month|year

if not PPLX_API_KEY:
    raise RuntimeError("PPLX_API_KEY is not set")

_client = OpenAI(api_key=PPLX_API_KEY, base_url=PPLX_BASE_URL)

# Small, fixed schema communicated to the model
MINI_SCHEMA_DOC = {
  "schema_version": "1.1",
  "symbol": "string (ticker, uppercased)",
  "headline": "string (<= 160 chars)",
  "recommendation": "buy | hold | sell | neutral",
  "conviction": "number in [0,1]",
  "rationale": [{"text": "string (<=25 words)", "source_idx": "int"}],  # <=4
  "highlights": [  # <=6
    {
      "label": "string",
      "value": "string | number | {low:number, high:number}",
      "as_of": "YYYY-MM-DD (optional)",
      "source_idx": "int"
    }
  ],
  "risks": [{"title": "string", "detail": "string", "source_idx": "int"}],  # <=4
  "citations": [{"title": "string", "url": "https://..."}]  # <=6
}

SYSTEM_PROMPT = f"""
You are a cautious retail investing explainer.
RULES:
- Use web search. Prefer primary sources (company IR/SEC) and reputable finance outlets.
- Be extractive and conservative. If unsure, omit.
- OUTPUT SIZE (strict): rationale ≤ 4 bullets, highlights ≤ 6 facts, risks ≤ 4, citations ≤ 6.
- No markdown, no prose outside JSON.
- Wrap the ONLY JSON object between these sentinels, exactly:
  {S}
  ...JSON here...
  {E}
""".strip()

def _user_payload(symbol: str) -> dict:
    sym = symbol.strip().upper()
    return {
        "task": "Analyze this investment symbol.",
        "holding": {"symbol": sym},
        "format": "Return ONLY a single JSON object between sentinels that matches this schema",
        "schema": MINI_SCHEMA_DOC,
        "notes": (
            "Keep strings short. Avoid control characters. "
            "Do not include citation markers like [1] in text; put sources in citations[]. "
        ),
    }

def _call_pplx(payload: str) -> Tuple[str, Optional[List[dict]]]:
    """One call to Perplexity via the OpenAI-compatible client. Returns (text, provider_citations?)."""
    # castedMessages = [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = _client.chat.completions.create(
        model=PPLX_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_user_payload(payload))},
        ],
        temperature=0.0,
        max_tokens=1600,
        # Perplexity-specific knobs live in extra_body
        extra_body={
            "search_recency_filter": SEARCH_RECENCY,  # day|week|month|year
            "return_images": False,
        },
    )
    text = (resp.choices[0].message.content or "").strip()
    # Perplexity sometimes attaches top-level citations; not in OpenAI spec, so guard it
    citations = getattr(resp, "citations", None)
    return text, citations
    
def analyze_investment_symbol_pplx(symbol: str) -> Dict[str, Any]:
    if not symbol or not symbol.strip():
        return {"error": "missing_symbol"}

    text, prov_citations = _call_pplx(symbol)

    try:
        data = extract_json(text)
    except Exception as e:
        return {"error": "parse_failed", "detail": str(e), "raw": text, "citations": prov_citations}

    # normalize/ensure citations exist (prefer model’s, else provider’s)
    data["citations"] = data.get("citations") or prov_citations
    return data