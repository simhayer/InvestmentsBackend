# services/helpers/ai/ai_forecaster.py  (rename your ai_forcaster.py if needed)
from __future__ import annotations
import json, datetime as dt
from typing import Any, Dict, List, Optional

def _clip(s: Optional[str], n: int) -> Optional[str]:
    if not s or not isinstance(s, str): return None
    return s[:n]

def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    # drop None/empty values to reduce tokens
    out = {}
    for k, v in d.items():
        if v is None: 
            continue
        if isinstance(v, (list, dict)) and not v:
            continue
        out[k] = v
    return out

def build_forecaster_payload(
    symbol: str,
    yahoo: Dict[str, Any],
    linkup_answer: Optional[str],
    linkup_items: List[Dict[str, str]],
    position: Optional[Dict[str, Any]] = None,
    horizon_days: int = 30,
) -> Dict[str, Any]:
    # keep only title+url; cap to 4 sources to keep prompt small
    citations = []
    for it in linkup_items[:4]:
        url = it.get("url")
        if url:
            citations.append({"title": _clip(it.get("title") or url, 140), "url": url})

    payload = {
        "as_of_utc": dt.datetime.utcnow().isoformat(),
        "symbol": symbol,
        "horizon_days": horizon_days,
        # send only the yahoo fields you’ll reference downstream
        "yahoo": _compact({
            "price": yahoo.get("current_price"),
            "day_change_pct": yahoo.get("day_change_pct"),
            "low_52w": yahoo.get("52_week_low"),
            "high_52w": yahoo.get("52_week_high"),
            "dist_to_52w_high_pct": yahoo.get("distance_from_52w_high_pct"),
            "dist_from_52w_low_pct": yahoo.get("distance_from_52w_low_pct"),
            "pe": yahoo.get("pe_ratio"),
            "yield_pct": yahoo.get("dividend_yield"),
            "beta": yahoo.get("beta"),
            "earnings_date_utc": yahoo.get("earnings_date_utc"),
            "ex_dividend_date_utc": yahoo.get("ex_dividend_date_utc"),
            "currency": yahoo.get("currency"),
        }),
        "position": _compact(position or {}),
        "news": {
            "summary": _clip(linkup_answer or "", 700),  # cap long Linkup answers
            "citations": citations,
        },
        # short schema + hard length caps
        "schema": {
            "confidence": "0.0-1.0",
            "scenarios": [
                {"name": "bull|base|bear", "prob": "0-1",
                 "drivers": ["..."], "watch": ["..."], "invalidations": ["..."], "urls": ["..."]}
            ],
            "events": [{"date": "ISO-8601", "event": "...", "why": "...", "url": "..."}],
            "notes": ["non-prescriptive, risk-aware tips"],
            "disclaimer": "string"
        },
        "constraints": [
            "Return VALID JSON only.",
            "Max 3 scenarios (bull, base, bear).",
            "Each list max 3 items; each item <= 15 words.",
            "Probabilities sum to 1. Use only provided yahoo/news facts.",
            "Cite only URLs from news.citations."
        ],
        "disclaimer": "This is educational information, not financial advice.",
    }
    return payload

def run_forecaster_analysis(llm: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a cautious market explainer. Map facts into forward scenarios."
        " Use ONLY provided fields; do not invent numbers/dates; no price targets."
        " Keep it concise per constraints and return VALID JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload)},
    ]

    # Bind a larger completion budget just for this call
    llm_bound = getattr(llm, "bind", lambda **kw: llm)(max_completion_tokens=1100)

    try:
        ai = llm_bound.invoke(messages)
        raw = ai.content if isinstance(ai.content, str) else str(ai.content)
        obj = json.loads(raw or "{}")
    except Exception as e:
        # If we hit a length/parse issue, retry once with extra brevity
        brief_messages = [
            {"role": "system", "content": system + " Be ultra brief."},
            {"role": "user", "content": json.dumps({**payload, "constraints": payload.get("constraints", []) + ["Be even shorter."]})},
        ]
        ai = llm_bound.invoke(brief_messages)
        raw = ai.content if isinstance(ai.content, str) else str(ai.content)
        try:
            obj = json.loads(raw or "{}")
        except Exception:
            obj = {}

    # Safe defaults if the model is still too long/invalid
    obj.setdefault("confidence", 0.5)
    obj.setdefault("scenarios", [
        {"name": "base", "prob": 0.5, "drivers": ["Limited info"], "watch": [], "invalidations": [], "urls": []},
        {"name": "bull", "prob": 0.25, "drivers": [], "watch": [], "invalidations": [], "urls": []},
        {"name": "bear", "prob": 0.25, "drivers": [], "watch": [], "invalidations": [], "urls": []},
    ])
    obj.setdefault("events", [])
    obj.setdefault("notes", ["Be mindful of event risk and concentration."])
    obj.setdefault("disclaimer", payload.get("disclaimer", "This is educational information, not financial advice."))
    return obj
