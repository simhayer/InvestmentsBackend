# services/ai/portfolio/facts_pack.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

from services.ai.portfolio.types import (
    HoldingClassification,
    HoldingRole,
    HoldingFlag,
)


class PortfolioTotals(BaseModel):
    currency: str = "USD"
    as_of: int

    market_value: float = 0.0
    total_cost_basis: float = 0.0

    day_pl: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_pl_pct: Optional[float] = None

    # Simple counts for quick UI
    holdings_count: int = 0
    priced_holdings_count: int = 0
    cost_basis_known_count: int = 0


class ConcentrationMetrics(BaseModel):
    top_1_weight_pct: float = 0.0
    top_3_weight_pct: float = 0.0
    top_5_weight_pct: float = 0.0
    hhi: float = 0.0  # Herfindahl–Hirschman index using weight fractions


class ExposureBreakdowns(BaseModel):
    by_type_weight_pct: Dict[str, float] = Field(default_factory=dict)
    by_institution_weight_pct: Dict[str, float] = Field(default_factory=dict)
    by_account_weight_pct: Dict[str, float] = Field(default_factory=dict)
    # Add sector later when you have it:
    by_sector_weight_pct: Dict[str, float] = Field(default_factory=dict)


class WinnersLosers(BaseModel):
    top_gainers: List[Dict[str, Any]] = Field(default_factory=list)  # {symbol, pl_pct, weight, pl}
    top_losers: List[Dict[str, Any]] = Field(default_factory=list)


class RoleSummary(BaseModel):
    core_weight_share_pct: float = 0.0
    risk_amplifier_weight_share_pct: float = 0.0
    satellite_weight_share_pct: float = 0.0

    core_symbols: List[str] = Field(default_factory=list)
    risk_symbols: List[str] = Field(default_factory=list)


class PortfolioFactsPack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    totals: PortfolioTotals
    concentration: ConcentrationMetrics
    exposures: ExposureBreakdowns
    roles: RoleSummary
    winners_losers: WinnersLosers

    data_quality_notes: List[str] = Field(default_factory=list)

    # Compact list for the LLM (don’t send huge tables)
    core_holdings_brief: List[Dict[str, Any]] = Field(default_factory=list)
    risk_holdings_brief: List[Dict[str, Any]] = Field(default_factory=list)
    satellite_brief: Dict[str, Any] = Field(default_factory=dict)


# -----------------------
# Helpers (pure python)
# -----------------------

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _norm_key(s: Optional[str], fallback: str = "Unknown") -> str:
        raw_t = (s or "").strip()
        t = raw_t
        if raw_t.lower() in ("cryptocurrency", "crypto", "cryptocurrencies"):
            t = "crypto"
        return t if t else fallback


def _pct(x: float) -> float:
    return round(x, 6)


def _hhi(weights_pct: List[float]) -> float:
    # weights_pct are in percent; convert to fractions
    w = [(max(0.0, float(x)) / 100.0) for x in weights_pct]
    return round(sum(x * x for x in w), 8)


def _sum_weights(items: List[Any]) -> float:
    return round(sum(_to_float(getattr(i, "weight", None), 0.0) for i in items), 6)


def _brief_row(h: Any) -> Dict[str, Any]:
    # Works with HoldingOut or dict-like; prefer attrs
    get = (lambda k, default=None: getattr(h, k, default)) if not isinstance(h, dict) else (lambda k, default=None: h.get(k, default))
    return {
        "symbol": str(get("symbol") or "").strip(),
        "type": get("type"),
        "weight_pct": round(_to_float(get("weight"), 0.0), 4),
        "value": round(_to_float(get("value") or get("current_value"), 0.0), 4),
        "unrealized_pl": round(_to_float(get("unrealized_pl"), 0.0), 4),
        "unrealized_pl_pct": (round(float(get("unrealized_pl_pct")), 4) if get("unrealized_pl_pct") is not None else None),
        "day_pl": round(_to_float(get("day_pl"), 0.0), 4),
    }


def build_portfolio_facts_pack(
    *,
    holdings: List[Any],  # List[HoldingOut] or dict-like
    classified_items: List[HoldingClassification],
    groups: Dict[str, List[HoldingClassification]],
    totals_payload: Dict[str, Any],  # from get_holdings_with_live_prices
) -> PortfolioFactsPack:
    """
    Build a compact facts pack for the LLM.
    The LLM should NEVER see the full raw holdings list if large.
    """

    currency = str(totals_payload.get("currency") or "USD")
    as_of = int(totals_payload.get("as_of") or 0)
    market_value = _to_float(totals_payload.get("market_value"), 0.0)

    # Totals from holdings (more reliable than just payload)
    total_cost = 0.0
    day_pl = 0.0
    unreal_pl = 0.0

    priced_count = 0
    cost_known_count = 0

    # Build quick lookups by id for classification flags
    class_by_id = {c.id: c for c in classified_items}

    # Exposures
    by_type = {}
    by_inst = {}
    by_acct = {}

    # Winners/Losers candidates (use unrealized_pl_pct)
    perf_rows = []

    for h in holdings or []:
        get = (lambda k, default=None: getattr(h, k, default)) if not isinstance(h, dict) else (lambda k, default=None: h.get(k, default))

        value = _to_float(get("value") or get("current_value"), 0.0)
        w = _to_float(get("weight"), 0.0)

        # totals
        day_pl += _to_float(get("day_pl"), 0.0)
        unreal_pl += _to_float(get("unrealized_pl"), 0.0)

        pat = get("purchase_amount_total")
        if pat is not None:
            total_cost += _to_float(pat, 0.0)
            cost_known_count += 1
        else:
            # Some users only have unit price and qty; you can extend later
            pass

        # counts
        if (get("current_price") is not None) and value > 0:
            priced_count += 1

        # exposures
        t = _norm_key(get("type"), "Unknown")
        inst = _norm_key(get("institution"), "Unknown")
        acct = _norm_key(get("account_name"), "Unknown")

        by_type[t] = by_type.get(t, 0.0) + w
        by_inst[inst] = by_inst.get(inst, 0.0) + w
        by_acct[acct] = by_acct.get(acct, 0.0) + w

        upl_pct = get("unrealized_pl_pct")
        perf_rows.append(
            {
                "symbol": str(get("symbol") or "").strip(),
                "pl_pct": float(upl_pct) if upl_pct is not None else None,
                "pl": _to_float(get("unrealized_pl"), 0.0),
                "weight": w,
            }
        )

    # Unrealized % (portfolio-level) — only if cost basis is meaningful
    unreal_pct = None
    if total_cost > 0.0:
        unreal_pct = round((unreal_pl / total_cost) * 100.0, 6)

    totals = PortfolioTotals(
        currency=currency,
        as_of=as_of,
        market_value=round(market_value, 6),
        total_cost_basis=round(total_cost, 6),
        day_pl=round(day_pl, 6),
        unrealized_pl=round(unreal_pl, 6),
        unrealized_pl_pct=unreal_pct,
        holdings_count=len(holdings or []),
        priced_holdings_count=priced_count,
        cost_basis_known_count=cost_known_count,
    )

    # Concentration from weights
    weights = sorted([_to_float(getattr(c, "weight"), 0.0) for c in classified_items], reverse=True)
    top1 = sum(weights[:1])
    top3 = sum(weights[:3])
    top5 = sum(weights[:5])

    concentration = ConcentrationMetrics(
        top_1_weight_pct=round(top1, 6),
        top_3_weight_pct=round(top3, 6),
        top_5_weight_pct=round(top5, 6),
        hhi=_hhi(weights),
    )

    exposures = ExposureBreakdowns(
        by_type_weight_pct={k: round(v, 6) for k, v in sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)},
        by_institution_weight_pct={k: round(v, 6) for k, v in sorted(by_inst.items(), key=lambda kv: kv[1], reverse=True)},
        by_account_weight_pct={k: round(v, 6) for k, v in sorted(by_acct.items(), key=lambda kv: kv[1], reverse=True)},
        by_sector_weight_pct={},  # later
    )

    drivers = groups.get("drivers", []) or []
    risk = groups.get("risk_amplifiers", []) or []
    sat = groups.get("satellites", []) or []

    roles = RoleSummary(
        core_weight_share_pct=round(_sum_weights(drivers), 6),
        risk_amplifier_weight_share_pct=round(_sum_weights(risk), 6),
        satellite_weight_share_pct=round(_sum_weights(sat), 6),
        core_symbols=[c.symbol for c in sorted(drivers, key=lambda x: (x.weight or 0.0), reverse=True)[:8]],
        risk_symbols=[c.symbol for c in sorted(risk, key=lambda x: (x.weight or 0.0), reverse=True)[:8]],
    )

    # Winners/Losers (use pl_pct but ignore None)
    perf_rows_clean = [r for r in perf_rows if r["pl_pct"] is not None]
    perf_rows_clean.sort(key=lambda r: r["pl_pct"], reverse=True)
    top_gainers = perf_rows_clean[:3]
    top_losers = list(reversed(perf_rows_clean[-3:])) if len(perf_rows_clean) >= 3 else perf_rows_clean[-3:]

    winners_losers = WinnersLosers(
        top_gainers=[
            {"symbol": r["symbol"], "unrealized_pl_pct": round(r["pl_pct"], 4), "weight_pct": round(r["weight"], 4), "unrealized_pl": round(r["pl"], 4)}
            for r in top_gainers
        ],
        top_losers=[
            {"symbol": r["symbol"], "unrealized_pl_pct": round(r["pl_pct"], 4), "weight_pct": round(r["weight"], 4), "unrealized_pl": round(r["pl"], 4)}
            for r in top_losers
        ],
    )

    # Data quality notes
    notes: List[str] = []
    missing_cost = [c for c in classified_items if HoldingFlag.missing_cost_basis in (c.flags or [])]
    missing_price = [c for c in classified_items if HoldingFlag.missing_price in (c.flags or [])]

    if totals.holdings_count > 0 and totals.priced_holdings_count < totals.holdings_count:
        notes.append(f"Live pricing missing for {totals.holdings_count - totals.priced_holdings_count} holding(s).")

    if totals.holdings_count > 0 and totals.cost_basis_known_count == 0:
        notes.append("Cost basis is missing for all holdings; unrealized P/L% may be incomplete.")
    elif missing_cost:
        notes.append(f"Cost basis missing for {len(missing_cost)} holding(s); portfolio P/L% may be partially estimated.")

    if missing_price:
        notes.append(f"Current price/value missing for {len(missing_price)} holding(s).")

    # Briefs for the LLM (don’t send everything)
    # Pick matching HoldingOut objects for core/risk by symbol (or id if you have it)
    # We’ll match by symbol since HoldingClassification has id but HoldingOut does too—use id if possible.
    hold_by_id = {}
    for h in holdings or []:
        get = (lambda k, default=None: getattr(h, k, default)) if not isinstance(h, dict) else (lambda k, default=None: h.get(k, default))
        hold_by_id[int(get("id"))] = h

    core_brief = []
    for c in sorted(drivers, key=lambda x: (x.weight or 0.0), reverse=True)[:8]:
        h = hold_by_id.get(c.id)
        row = _brief_row(h) if h is not None else {"symbol": c.symbol, "weight_pct": c.weight, "value": c.value}
        row["role"] = "driver"  # clearer label for UI, even if field name stays core_holdings_brief
        row["flags"] = [f.value for f in (c.flags or [])]
        core_brief.append(row)

    risk_brief = []
    for c in sorted(risk, key=lambda x: (x.weight or 0.0), reverse=True)[:8]:
        h = hold_by_id.get(c.id)
        row = _brief_row(h) if h is not None else {"symbol": c.symbol, "weight_pct": c.weight, "value": c.value}
        row["role"] = "risk_amplifier"
        row["flags"] = [f.value for f in (c.flags or [])]
        risk_brief.append(row)

    sat_share = round(_sum_weights(sat), 6)
    sat_count = len(sat)
    satellite_brief = {
        "count": sat_count,
        "weight_share_pct": sat_share,
        "note": "Satellite positions are smaller and summarized to keep analysis focused.",
    }

    return PortfolioFactsPack(
        totals=totals,
        concentration=concentration,
        exposures=exposures,
        roles=roles,
        winners_losers=winners_losers,
        data_quality_notes=notes,
        core_holdings_brief=core_brief,
        risk_holdings_brief=risk_brief,
        satellite_brief=satellite_brief,
    )
