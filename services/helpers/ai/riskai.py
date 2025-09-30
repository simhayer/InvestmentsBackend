# services/portfolio/risk_ai.py
from typing import Any, Dict, List
import json
from services.helpers.ai.json_helpers import S, E, extract_json  # you already have these

from services.helpers.ai.ai_config import _client, PPLX_MODEL

AI_RISK_SCHEMA = {
  "overall_title": "string (e.g., 'Moderate')",
  "overall_blurb": "string (<= 280 chars)",
  "diversification": "string (<= 200 chars)",
  "volatility": "string (<= 200 chars)",
  "suggestions": ["string (<= 120 chars)"]  # 3–5 items
}

AI_RISK_SYSTEM = f"""
You turn portfolio risk numbers into a clear summary and concrete suggestions.
Rules:
- Use only the numbers provided by the user payload; do not invent metrics.
- Be concise, practical, and neutral.
- Output exactly ONE JSON object, wrapped by these sentinels:
{S}
...json...
{E}
- Match this schema: {json.dumps(AI_RISK_SCHEMA)}
""".strip()

def generate_ai_risk_summary(risk: Dict[str, Any], summary: Dict[str, Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "task": "Summarize portfolio risk and propose actions based ONLY on provided inputs.",
        "inputs": {
            "overall_score": risk["overall"]["score"],
            "overall_level": risk["overall"]["level"],                # Low/Moderate/High
            "div_score": risk["diversification"]["score"],
            "hhi": risk["diversification"]["hhi"],
            "n_eff": risk["diversification"]["n_eff"],
            "vol_score": risk["volatility"]["score"],
            "est_sigma_annual_pct": risk["volatility"]["est_vol_annual_pct"],
            "top_position": summary["top_position"],
            "by_sector": summary["by_sector"],
            "by_country": summary["by_country"],
            "positions_sample": positions[:8],  # small sample for color; contains {symbol, weight, sector, country}
        }
    }

    resp = _client.chat.completions.create(
        model=PPLX_MODEL,
        temperature=0.2,
        max_tokens=500,
        messages=[
            {"role": "system", "content": AI_RISK_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        extra_body={"search_recency_filter": "month", "return_images": False},
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        return extract_json(text)
    except Exception:
        return {
            "overall_title": risk["overall"]["level"],
            "overall_blurb": "Balanced profile with room for optimization.",
            "diversification": "Diversification is the main driver of safety; consider broadening if concentrated.",
            "volatility": "Estimated volatility is derived from asset-class proxies.",
            "suggestions": ["Review position sizing", "Add non-US exposure", "Mix in lower-vol assets"],
        }
