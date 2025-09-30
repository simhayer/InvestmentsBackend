# services/portfolio/risk.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

# 1) Very small classification layer. Extend as needed.
ETF_CLASS = {
    "SPY": "us_equity", "VOO": "us_equity", "VTI": "us_equity",
    "VEA": "dev_ex_us_equity", "IEFA": "dev_ex_us_equity",
    "VWO": "em_equity", "EEM": "em_equity",
    "BND": "bond", "AGG": "bond", "TLT": "bond",
    "GLD": "gold", "IAU": "gold",
    "BTC": "crypto", "ETH": "crypto",
    "EWZ": "em_equity",  # Brazil
}
# Annualized vol assumptions (very rough)
CLASS_SIGMA = {
    "us_equity": 0.18,
    "dev_ex_us_equity": 0.16,
    "em_equity": 0.22,
    "bond": 0.05,
    "gold": 0.15,
    "crypto": 0.80,
    "equity_default": 0.18,
    "other": 0.20,
}

def _classify(symbol: str, sector: str | None) -> Tuple[str, float]:
    s = symbol.upper().strip()
    asset_class = ETF_CLASS.get(s)
    if not asset_class:
        # Heuristic: anything that looks like a stock defaults to equity
        if sector and sector.lower() in {"financials","energy","materials","industrials",
                                         "healthcare","consumer discretionary","consumer staples",
                                         "utilities","real estate","communication services",
                                         "information technology","tech","technology","it"}:
            asset_class = "us_equity"
        elif s in {"BTC","ETH"}:
            asset_class = "crypto"
        else:
            # Unknown: treat as equity-like
            asset_class = "equity_default"
    sigma = CLASS_SIGMA.get(asset_class, CLASS_SIGMA["other"])
    return asset_class, sigma

def _score_diversification(hhi: float) -> tuple[float, float]:
    """
    n_eff = 1/HHI; map n_eff∈[1..25] to score∈[0..100], higher = safer (more diversified).
    """
    if hhi <= 0:  # guard
        return 100.0, float("inf")
    n_eff = 1.0 / hhi
    n_cap = min(n_eff, 25.0)
    score = max(0.0, min(100.0, (n_cap - 1.0) / (25.0 - 1.0) * 100.0))
    return score, n_eff

def _score_volatility(positions: list[dict]) -> tuple[float, float, list[dict], list[str]]:
    """
    Estimate portfolio vol as sqrt(sum (w_i*sigma_i)^2)  (assumes zero correlation).
    Map vol% from [5%..40%] to safety score 100→0 (higher vol = lower safety).
    """
    details, notes = [], []
    quad_sum = 0.0
    for p in positions:
        sym = p["symbol"]
        w   = float(p["weight"])
        sec = (p.get("sector") or "").strip() or None
        cls, sigma = _classify(sym, sec)
        quad_sum += (w * sigma) ** 2
        details.append({"symbol": sym, "class": cls, "weight": w, "sigma_pct": round(sigma*100, 1)})

    est_sigma = math.sqrt(quad_sum)  # annualized %
    # Scale to safety score
    low, high = 0.05, 0.40
    clamped = max(low, min(high, est_sigma))
    safety = (high - clamped) / (high - low) * 100.0
    return safety, est_sigma * 100.0, details, notes

def compute_risk(summary: dict, positions: list[dict]) -> dict[str, Any]:
    """
    Inputs:
      summary: dict with 'hhi'
      positions: [{symbol, weight, sector?}, ...]   (weights sum to 1)
    Returns a 'risk' object you can attach to your API response.
    """
    div_score, n_eff = _score_diversification(float(summary.get("hhi", 0.0)))
    vol_score, est_vol_pct, class_details, notes = _score_volatility(positions)

    overall = round((div_score + vol_score) / 2.0, 1)
    if overall >= 75:
        level = "Low"
    elif overall >= 55:
        level = "Moderate"
    else:
        level = "High"

    return {
        "overall": {"score": overall, "level": level},
        "diversification": {
            "score": round(div_score, 1),
            "n_eff": round(n_eff, 2) if math.isfinite(n_eff) else None,
            "hhi": round(float(summary.get("hhi", 0.0)), 3),
            "label": "Higher is better (more independent positions).",
        },
        "volatility": {
            "score": round(vol_score, 1),
            "est_vol_annual_pct": round(est_vol_pct, 1),
            "label": "Based on asset-class proxies (no correlations).",
        },
        "details": {
            "classes": class_details,  # [{symbol, class, weight, sigma_pct}]
            "notes": notes,
        },
        "disclaimer": "Educational; not investment advice. Volatility/HHI are approximations.",
    }
