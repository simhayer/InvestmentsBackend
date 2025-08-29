# services/helpers/ai/forecast_post.py
from typing import Any, Dict, List

def normalize_forecast(
    obj: Dict[str, Any],
    citations: List[Dict[str, str]],
    yahoo: Dict[str, Any],
    position: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # 1) normalize confidence
    try:
        obj["confidence"] = float(obj.get("confidence", 0.5))
    except Exception:
        obj["confidence"] = 0.5

    # 2) normalize scenario probabilities and clamp 0..1
    scens = obj.get("scenarios") or []
    total = 0.0
    for s in scens:
        try:
            p = float(s.get("prob", 0.0))
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        s["prob"] = p
        total += p
    if total > 0:
        for s in scens:
            s["prob"] = round(s["prob"] / total, 3)
    else:
        scens = [
            {"name": "base", "prob": 0.5, "drivers": [], "watch": [], "invalidations": [], "urls": []},
            {"name": "bull", "prob": 0.25, "drivers": [], "watch": [], "invalidations": [], "urls": []},
            {"name": "bear", "prob": 0.25, "drivers": [], "watch": [], "invalidations": [], "urls": []},
        ]

    # 3) keep scenario URLs within allowed citations only; ensure ≥1
    allowed = [c["url"] for c in citations if c.get("url")]
    for s in scens:
        urls = [u for u in (s.get("urls") or []) if u in allowed]
        if not urls and allowed:
            urls = allowed[:1]
        s["urls"] = urls[:2]
    obj["scenarios"] = scens

    # 4) compute a simple risk level from beta and/or position weight
    beta = yahoo.get("beta")
    risk = "moderate"
    if isinstance(beta, (int, float)):
        risk = "low" if beta < 0.7 else "moderate" if beta < 1.2 else "high"
    weight = (position or {}).get("weight_pct")
    if isinstance(weight, (int, float)) and weight >= 20:
        # bump risk one notch for concentration
        risk = "high" if risk == "moderate" else risk
    obj["risk_level"] = risk

    # 5) add deterministic events from Yahoo if missing
    events = obj.get("events") or []
    exd = yahoo.get("ex_dividend_date_utc")
    if exd:
        events.append({"date": exd, "event": "Ex-dividend date", "why": "Dividend capture / near-term volatility"})
    ed = yahoo.get("earnings_date_utc")
    if ed:
        events.append({"date": ed, "event": "Earnings", "why": "Guidance and provision trends can move the stock"})
    obj["events"] = events

    return obj
