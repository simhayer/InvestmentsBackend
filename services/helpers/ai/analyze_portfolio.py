# helpers/ai/analyze_portfolio.py
from __future__ import annotations

import json
import datetime as dt
from typing import Any, Dict, List
from collections import defaultdict
from utils.common_helpers import round, pct_change, safe_float

def _position_value(pos: Dict[str, Any]) -> float:
    """
    Prefer explicit `value`; else quantity*current_price; else 0.
    """
    val = safe_float(pos.get("value"))
    if val is not None:
        return val
    q = safe_float(pos.get("quantity")) or 0.0
    cp = safe_float(pos.get("current_price")) or 0.0
    return q * cp

def _cost_basis(pos: Dict[str, Any]) -> float:
    q = safe_float(pos.get("quantity")) or 0.0
    pp = safe_float(pos.get("purchase_price")) or 0.0
    return q * pp

def analyze_portfolio(holdings: List[Dict[str, Any]], llm: Any) -> str:
    """
    Portfolio-level analysis only (rating, risks, diversification).
    Returns VALID JSON string with this schema:

    {
      "as_of_utc": ISO-8601 string,
      "totals": {
        "market_value": number,
        "cost_basis": number,
        "pnl_abs": number,
        "pnl_pct": number
      },
      "exposures": {
        "by_type": [{"type": str, "weight_pct": number}],
        "by_currency": [{"currency": str, "weight_pct": number}],
        "by_institution": [{"institution": str, "weight_pct": number}]
      },
      "top_positions": [{"symbol": str, "weight_pct": number, "pnl_pct": number}],
      "concentration": {
        "hh_index": number,              # Herfindahl-Hirschman Index (0-1)
        "top_1_weight_pct": number,
        "top_3_weight_pct": number,
        "top_5_weight_pct": number
      },
      "rating": "strong|balanced|concentrated|risky|needs_rebalance",
      "risk_level": "low|moderate|high",
      "diversification_score": number,   # 0-100 (higher = better diversified)
      "rationale": str,                  # brief paragraph
      "suggestions": [str, ...],         # 3-5 bullets
      "data_notes": [str, ...],
      "disclaimer": "This is educational information, not financial advice."
    }
    """
    as_of = dt.datetime.utcnow().isoformat()

    # ---- Aggregate core numbers
    positions: List[Dict[str, Any]] = []
    total_mv = 0.0
    total_cb = 0.0

    for h in holdings:
        mv = _position_value(h)
        cb = _cost_basis(h)
        total_mv += mv
        total_cb += cb
        positions.append({
            "symbol": (h.get("symbol") or "").upper(),
            "name": h.get("name"),
            "type": (h.get("type") or "stock").lower(),
            "currency": h.get("currency") or "USD",
            "institution": h.get("institution") or "N/A",
            "market_value": mv,
            "cost_basis": cb,
            "pnl_abs": mv - cb,
            "pnl_pct": pct_change(mv, cb),
        })

    pnl_abs = total_mv - total_cb
    pnl_pct = pct_change(total_mv, total_cb)

    # ---- Weights & exposures
    def _weights_by(key: str) -> List[Dict[str, Any]]:
        bucket = defaultdict(float)
        if total_mv > 0:
            for p in positions:
                bucket[str(p.get(key) or "N/A")] += p["market_value"]
        items = [{"label": k, "weight_pct": (v / total_mv * 100.0) if total_mv > 0 else 0.0} for k, v in bucket.items()]
        # sort desc
        items.sort(key=lambda x: x["weight_pct"], reverse=True)
        # remap label->typed key name
        out = []
        for it in items:
            if key == "type":
                out.append({"type": it["label"], "weight_pct": round(it["weight_pct"], 4)})
            elif key == "currency":
                out.append({"currency": it["label"], "weight_pct": round(it["weight_pct"], 4)})
            else:
                out.append({"institution": it["label"], "weight_pct": round(it["weight_pct"], 4)})
        return out

    by_type = _weights_by("type")
    by_currency = _weights_by("currency")
    by_institution = _weights_by("institution")

    # ---- Top positions and concentration (Herfindahl)
    weights = []
    for p in positions:
        w = (p["market_value"] / total_mv * 100.0) if total_mv > 0 else 0.0
        weights.append({"symbol": p["symbol"], "weight_pct": w, "pnl_pct": p["pnl_pct"]})
    weights.sort(key=lambda x: x["weight_pct"], reverse=True)
    top_positions = [{"symbol": w["symbol"], "weight_pct": round(w["weight_pct"], 4), "pnl_pct": round(w["pnl_pct"], 4)} for w in weights[:5]]

    # Herfindahl-Hirschman Index using fractional weights (0-1)
    hh_index = 0.0
    for w in weights:
        f = (w["weight_pct"] or 0.0) / 100.0
        hh_index += f * f

    top_1 = round(weights[0]["weight_pct"], 4) if weights else 0.0
    top_3 = round(sum(w["weight_pct"] for w in weights[:3]), 4) if weights else 0.0
    top_5 = round(sum(w["weight_pct"] for w in weights[:5]), 4) if weights else 0.0

    # ---- Heuristic diversification score (0-100)
    # Penalize concentration (HHI), large top weights, and single-currency exposure
    hhi_penalty = min(hh_index, 1.0) * 60.0           # up to -60
    top_penalty = (max(0.0, (top_3 or 0.0) - 40.0)) * 0.5  # -0.5 per % above 40
    currency_penalty = 0.0 if len(by_currency) >= 2 and by_currency[0]["weight_pct"] <= 80 else 10.0
    diversification_score = max(0.0, 100.0 - (hhi_penalty * 100 if hhi_penalty <= 1 else hhi_penalty) - top_penalty - currency_penalty)

    # Normalize hhi_penalty used above: ensure we put the raw 0-1 HHI in output
    # (we already computed hh_index as 0-1)

    # ---- Build compact payload for the LLM (numbers only; no external fetches)
    payload = {
        "as_of_utc": as_of,
        "totals": {
            "market_value": round(total_mv, 4),
            "cost_basis": round(total_cb, 4),
            "pnl_abs": round(pnl_abs, 4),
            "pnl_pct": round(pnl_pct, 4),
        },
        "exposures": {
            "by_type": by_type,
            "by_currency": by_currency,
            "by_institution": by_institution,
        },
        "top_positions": top_positions,
        "concentration": {
            "hh_index": round(hh_index, 6),
            "top_1_weight_pct": round(top_1, 4),
            "top_3_weight_pct": round(top_3, 4),
            "top_5_weight_pct": round(top_5, 4),
        },
        "diversification_score": round(diversification_score, 4),
        "must_include": [
            "Overall portfolio rating (strong|balanced|concentrated|risky|needs_rebalance).",
            "Risk level (low|moderate|high).",
            "Short rationale referencing totals, top weights, and currency/type exposure.",
            "3-5 actionable suggestions (e.g., trim oversized positions, diversify currency, rebalance).",
            "Data notes + disclaimer."
        ],
        "schema": {
            "as_of_utc": "ISO-8601 string",
            "totals": {"market_value": "number", "cost_basis": "number", "pnl_abs": "number", "pnl_pct": "number"},
            "exposures": {"by_type": "array", "by_currency": "array", "by_institution": "array"},
            "top_positions": "array",
            "concentration": {"hh_index": "number (0-1)", "top_1_weight_pct": "number", "top_3_weight_pct": "number", "top_5_weight_pct": "number"},
            "rating": "one of: strong | balanced | concentrated | risky | needs_rebalance",
            "risk_level": "one of: low | moderate | high",
            "diversification_score": "0-100",
            "rationale": "string, concise paragraph",
            "suggestions": "array of 3-5 strings",
            "data_notes": "array of strings",
            "disclaimer": "string",
        },
        "disclaimer": "This is educational information, not financial advice.",
    }

    system = (
        "You are a cautious retail investing explainer. "
        "Use ONLY the provided numbers; do not invent data or give personalized advice. "
        "Respond with VALID JSON that matches the requested schema. Keep it concise."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload)},
    ]

    # ---- Call the model & parse; harden with fallback
    
    ai = llm.invoke(messages)
    try:
        raw = ai.content
        if isinstance(raw, list):
            raw = "".join(str(p) for p in raw)
        if not isinstance(raw, str):
            raw = str(raw)
        obj = json.loads(raw)
    except Exception:
        obj = {
            "as_of_utc": as_of,
            "totals": payload["totals"],
            "exposures": payload["exposures"],
            "top_positions": payload["top_positions"],
            "concentration": payload["concentration"],
            "rating": "balanced",
            "risk_level": "moderate",
            "diversification_score": payload["diversification_score"],
            "rationale": "Automatic fallback: could not parse model output. Based on provided weights and totals, the portfolio appears reasonably balanced but may benefit from periodic rebalancing.",
            "suggestions": [
                "Set target weights and rebalance if any top positions exceed target by >5%.",
                "Diversify currency exposure if >80% in a single currency.",
                "Review position sizing for the top 3 holdings.",
            ],
            "data_notes": ["No external data fetched; analysis based on your stored holdings."],
            "disclaimer": "This is educational information, not financial advice.",
        }

    # Ensure required fields exist + patch objective numbers from our calculations
    obj.setdefault("as_of_utc", as_of)
    obj.setdefault("totals", payload["totals"])
    obj.setdefault("exposures", payload["exposures"])
    obj.setdefault("top_positions", payload["top_positions"])
    obj.setdefault("concentration", payload["concentration"])
    obj.setdefault("diversification_score", payload["diversification_score"])
    obj.setdefault("data_notes", ["No external data fetched; analysis based on your stored holdings."])
    obj.setdefault("disclaimer", "This is educational information, not financial advice.")

    # Keep categorical fields sane
    valid_ratings = {"strong", "balanced", "concentrated", "risky", "needs_rebalance"}
    rating = str(obj.get("rating", "balanced")).lower().strip()
    obj["rating"] = rating if rating in valid_ratings else "balanced"

    valid_risk = {"low", "moderate", "high"}
    risk = str(obj.get("risk_level", "moderate")).lower().strip()
    obj["risk_level"] = risk if risk in valid_risk else "moderate"

    # Return as JSON string (API-friendly)
    return json.dumps(obj, ensure_ascii=False)
