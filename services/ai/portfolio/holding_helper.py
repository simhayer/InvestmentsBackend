from typing import Any, Dict, List
from services.ai.portfolio.types import (
    HoldingClassification, HoldingFlag, ClassifyConfig
)

def classify_holdings(
    holdings: List[Any],
    *,
    market_value: float | None = None,
    config: ClassifyConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or ClassifyConfig()

    def get(h, k, default=None):
        return getattr(h, k, default) if not isinstance(h, dict) else h.get(k, default)

    def to_float(x, default=None):
        try:
            return float(x) if x is not None else default
        except Exception:
            return default

    def normalize_asset_type_label(t: str | None) -> str:
        x = (t or "").strip().lower()
        if x in ("crypto", "cryptocurrency", "cryptocurrencies"):
            return "crypto"
        if x in ("equity", "stock", "shares"):
            return "equity"
        if x in ("etf", "fund"):
            return "etf"
        return x or "unknown"

    # compute market value if needed
    mv = float(market_value or 0.0)
    if mv <= 0:
        mv = 0.0
        for h in holdings or []:
            mv += to_float(get(h, "value") or get(h, "current_value"), 0.0) or 0.0

    # normalize holdings with weights
    norm = []
    for h in holdings or []:
        value = to_float(get(h, "value") or get(h, "current_value"), 0.0) or 0.0
        weight = to_float(get(h, "weight"))
        if weight is None and mv > 0:
            weight = (value / mv) * 100.0

        norm.append({
            "h": h,
            "id": int(get(h, "id")),
            "symbol": str(get(h, "symbol") or "").strip(),
            "type": get(h, "type"),
            "type_norm": normalize_asset_type_label(get(h, "type")),
            "value": value,
            "weight": float(weight or 0.0),
            "unrealized_pl": to_float(get(h, "unrealized_pl")),
            "unrealized_pl_pct": to_float(get(h, "unrealized_pl_pct")),
            "purchase_amount_total": to_float(get(h, "purchase_amount_total")),
            "purchase_unit_price": to_float(get(h, "purchase_unit_price")),
            "current_price": to_float(get(h, "current_price")),
        })

    # --- DRIVER SELECTION (cumulative weight) ---
    norm_sorted = sorted(norm, key=lambda x: x["weight"], reverse=True)
    driver_ids = set()
    cum = 0.0
    for x in norm_sorted:
        if len(driver_ids) >= cfg.top_n_core:
            break
        if cum >= 85.0:  # target coverage, tune 80–90
            break
        driver_ids.add(x["id"])
        cum += x["weight"]

    items: List[HoldingClassification] = []
    for n in norm:
        flags: List[HoldingFlag] = []
        reasons: List[str] = []

        weight = n["weight"]
        upl_pct = n["unrealized_pl_pct"]
        type_norm = n["type_norm"]

        # Data quality flags
        if (n["current_price"] is None) or (n["value"] <= 0):
            flags.append(HoldingFlag.missing_price)
            reasons.append("Missing or zero current price/value.")
        if (n["purchase_amount_total"] is None) and (n["purchase_unit_price"] is None):
            flags.append(HoldingFlag.missing_cost_basis)
            reasons.append("Missing cost basis (purchase totals/unit price).")

        # Driver tag (not a role)
        is_driver = (n["id"] in driver_ids) or (weight >= cfg.core_weight_pct)
        if is_driver:
            flags.append(HoldingFlag.concentration)
            reasons.append(f"Driver holding (weight {weight:.2f}%).")

        # Risk amplifier tags
        is_risk = False
        if type_norm in set(cfg.risk_types):
            flags.append(HoldingFlag.high_volatility_type)
            reasons.append(f"High-volatility type: {type_norm}.")
            is_risk = True

        if upl_pct is not None and upl_pct <= cfg.big_loser_pct:
            flags.append(HoldingFlag.big_loser)
            reasons.append(f"Large unrealized loss: {upl_pct:.2f}%.")
            is_risk = True

        if weight <= cfg.tiny_weight_pct:
            flags.append(HoldingFlag.tiny_position)
            reasons.append(f"Small position: {weight:.2f}%.")

        # score (optional)
        score = 0.0
        score += min(weight / 5.0, 10.0)
        if HoldingFlag.big_loser in flags: score += 3.0
        if HoldingFlag.high_volatility_type in flags: score += 2.0

        items.append(
            HoldingClassification(
                id=n["id"],
                symbol=n["symbol"],
                type=n["type"],
                value=n["value"],
                weight=weight,
                unrealized_pl=n["unrealized_pl"],
                unrealized_pl_pct=upl_pct,
                is_driver=is_driver,
                is_risk_amplifier=is_risk,
                flags=sorted(list(set(flags)), key=lambda f: f.value),
                reasons=reasons,
                score=round(score, 3),
            )
        )

    # --- GROUPS (now independent) ---
    drivers = [c for c in items if c.is_driver]
    risk_amplifiers = [c for c in items if c.is_risk_amplifier]
    satellites = [c for c in items if (not c.is_driver)]

    summary = {
        "market_value": mv,
        "counts": {
            "drivers": len(drivers),
            "risk_amplifiers": len(risk_amplifiers),
            "satellites": len(satellites),
        },
        "weight_share_pct": {
            "drivers": round(sum(c.weight or 0.0 for c in drivers), 6),
            "risk_amplifiers": round(sum(c.weight or 0.0 for c in risk_amplifiers), 6),
            "satellites": round(sum(c.weight or 0.0 for c in satellites), 6),
        },
        "driver_symbols": [c.symbol for c in sorted(drivers, key=lambda x: x.weight or 0.0, reverse=True)],
        "risk_symbols": [c.symbol for c in sorted(risk_amplifiers, key=lambda x: x.weight or 0.0, reverse=True)],
    }

    return {
        "items": items,
        "groups": {
            "drivers": drivers,
            "risk_amplifiers": risk_amplifiers,
            "satellites": satellites,
        },
        "summary": summary,
    }
