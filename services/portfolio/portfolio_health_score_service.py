# services/portfolio/portfolio_health_score_v2_service.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Literal
from collections import defaultdict

from sqlalchemy.orm import Session

from services.finnhub.finnhub_service import FinnhubService
from services.holding_service import get_holdings_with_live_prices
from services.finnhub.finnhub_profile_cache import fetch_profiles_cached
from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached

from schemas.portfolio_health_score import PortfolioHealthScoreResponse, HealthSubscores

Baseline = Literal["balanced", "growth", "conservative"]


def _to_float(x: Any) -> float:
    try:
        return float(x) if x is not None else 0.0
    except Exception:
        return 0.0


def _pct(n: float, d: float) -> float:
    return round((n / d * 100.0), 6) if d else 0.0


def _grade(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _normalize_type(t: str | None) -> str:
    tt = (t or "").strip().lower()
    if tt in ("equity", "stock", "stocks"):
        return "equity"
    if tt in ("etf", "fund"):
        return "etf"
    if tt in ("crypto", "cryptocurrency"):
        return "crypto"
    if tt in ("cash", "currency", "fx"):
        return "cash"
    if not tt:
        return "unknown"
    return tt


def _hhi(weights_pct: List[float]) -> float:
    return round(sum((w / 100.0) ** 2 for w in weights_pct), 6)


def _score_sector_region(
    sector_weights: Dict[str, float],
    region_weights: Dict[str, float],
    insights: List[str],
    suggestions: List[str],
) -> int:
    score = 25

    if sector_weights:
        top_sector, top_sector_w = max(sector_weights.items(), key=lambda kv: kv[1])
        if top_sector_w >= 50:
            score -= 10
            insights.append(f"Sector concentration is high: {top_sector} is {top_sector_w:.1f}%.")
            suggestions.append("Reduce your top sector exposure by adding other sectors or broad ETFs.")
        elif top_sector_w >= 35:
            score -= 5
            insights.append(f"Top sector is {top_sector} at {top_sector_w:.1f}% (moderate concentration).")
        else:
            insights.append(f"Sector exposure looks balanced (top sector {top_sector} at {top_sector_w:.1f}%).")

    if region_weights:
        top_region, top_region_w = max(region_weights.items(), key=lambda kv: kv[1])
        if top_region_w >= 90:
            score -= 8
            insights.append(f"Region exposure is very concentrated: {top_region} is {top_region_w:.1f}%.")
            suggestions.append("Add international exposure if you want to reduce single-country risk.")
        elif top_region_w >= 75:
            score -= 4
            insights.append(f"Region exposure is concentrated: {top_region} is {top_region_w:.1f}%.")
        else:
            insights.append(f"Region exposure looks balanced (top region {top_region} at {top_region_w:.1f}%).")

    return max(0, min(25, int(round(score))))


def _score_quality(
    equity_quality_weighted: float,
    covered_equity_weight: float,
    insights: List[str],
    suggestions: List[str],
) -> int:
    score = 15

    # If we only covered a tiny slice of the portfolio, don’t pretend we know.
    if covered_equity_weight <= 5:
        score -= 3
        insights.append("Limited fundamentals coverage; quality score is best-effort.")
        return max(0, min(15, int(round(score))))

    if equity_quality_weighted >= 0.70:
        insights.append("Quality tilt looks strong (profitability/financial health metrics are solid).")
    elif equity_quality_weighted >= 0.50:
        score -= 3
        insights.append("Quality tilt looks mixed based on available fundamentals.")
        suggestions.append("If you want higher quality tilt, increase exposure to profitable, lower-debt companies or quality ETFs.")
    else:
        score -= 7
        insights.append("Quality tilt looks weak based on available fundamentals.")
        suggestions.append("Consider balancing with quality ETFs or reducing weaker-quality holdings.")

    return max(0, min(15, int(round(score))))


def _quality_from_normalized(norm: Dict[str, Any]) -> float:
    if not isinstance(norm, dict):
        return 0.0

    opm = norm.get("operating_margin")
    gm = norm.get("gross_margin")
    dte = norm.get("debt_to_equity")
    fcf = norm.get("free_cash_flow")

    score = 0.0
    denom = 0.0

    # profitability
    if isinstance(opm, (int, float)):
        denom += 1
        score += 1.0 if opm >= 10 else 0.6 if opm >= 5 else 0.2
    elif isinstance(gm, (int, float)):
        denom += 1
        score += 1.0 if gm >= 40 else 0.6 if gm >= 25 else 0.2

    # leverage
    if isinstance(dte, (int, float)):
        denom += 1
        score += 1.0 if dte <= 1.0 else 0.6 if dte <= 2.0 else 0.2

    # fcf proxy
    if isinstance(fcf, (int, float)):
        denom += 1
        score += 1.0 if fcf > 0 else 0.2

    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, score / denom))


def _country_to_region(country: str) -> str:
    c = (country or "").strip().upper()
    if c in ("US", "USA", "UNITED STATES"):
        return "US"
    if c in ("CA", "CAN", "CANADA"):
        return "Canada"
    if not c:
        return "Unknown"
    return "International"


async def build_portfolio_health_score(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    *,
    baseline: Baseline = "balanced",
    fundamentals_top_n: int = 10,  # ✅ only fetch fundamentals for top N equities by weight
) -> PortfolioHealthScoreResponse:
    data = await get_holdings_with_live_prices(
        user_id=user_id,
        db=db,
        finnhub=finnhub,
        currency=currency,
        top_only=False,
        top_n=5,
        include_weights=True,
    )

    items = data.get("items") or []
    as_of = int(data.get("as_of") or 0)
    curr = str(data.get("currency") or currency).upper()
    market_value = _to_float(data.get("market_value"))

    insights: List[str] = []
    suggestions: List[str] = []
    notes: List[str] = []

    if not items:
        return PortfolioHealthScoreResponse(
            user_id=user_id,
            currency=curr,
            as_of=as_of,
            score=0,
            grade="F",
            baseline=baseline,
            subscores=HealthSubscores(diversification=0, risk_balance=0, sector_region=0, quality=0),
            sector_weights_pct={},
            region_weights_pct={},
            insights=["No holdings found."],
            suggestions=["Add holdings to calculate portfolio health."],
            notes=[],
        )

    # -----------------------
    # Phase 1 quick metrics
    # -----------------------
    weights_pct = [max(0.0, _to_float(getattr(it, "weight", 0.0))) for it in items]
    weights_sorted = sorted(weights_pct, reverse=True)
    top1 = round(sum(weights_sorted[:1]), 6)
    top5 = round(sum(weights_sorted[:5]), 6)
    hhi = _hhi(weights_pct)

    # Diversification (0..60) - simple, tune later
    div_score = 60
    if len(items) < 5:
        div_score -= 18
    if top1 >= 25:
        div_score -= 10
    if top5 >= 65:
        div_score -= 8
    if hhi > 0.18:
        div_score -= 8
    div_score = max(0, min(60, int(round(div_score))))

    # Risk balance (0..40) - type mix (phase 1)
    by_type_value = defaultdict(float)
    for it in items:
        t = _normalize_type(getattr(it, "type", None))
        v = _to_float(getattr(it, "current_value", None) or getattr(it, "value", None))
        by_type_value[t] += v

    type_weights = {k: _pct(v, market_value) for k, v in by_type_value.items()} if market_value else {}
    crypto_w = float(type_weights.get("crypto", 0.0))
    risk_score = 40
    if crypto_w >= 25:
        risk_score -= 14
    elif crypto_w >= 10:
        risk_score -= 6
    risk_score = max(0, min(40, int(round(risk_score))))

    # -----------------------
    # Build symbol/value maps
    # -----------------------
    value_by_symbol: Dict[str, float] = {}
    type_by_symbol: Dict[str, str] = {}

    for it in items:
        sym = (getattr(it, "symbol", "") or "").strip().upper()
        if not sym:
            continue
        t = _normalize_type(getattr(it, "type", None))
        v = _to_float(getattr(it, "current_value", None) or getattr(it, "value", None))
        value_by_symbol[sym] = value_by_symbol.get(sym, 0.0) + v
        type_by_symbol[sym] = t

    # -----------------------
    # Sector + Region exposure
    # -----------------------
    equity_symbols = [s for s, t in type_by_symbol.items() if t == "equity"]
    profiles = await fetch_profiles_cached(finnhub, equity_symbols, max_concurrency=6)

    sector_value = defaultdict(float)
    region_value = defaultdict(float)

    for sym, v in value_by_symbol.items():
        t = type_by_symbol.get(sym, "unknown")
        if t != "equity":
            # bucket non-equities into their own groups
            sector_value[t.upper()] += v
            region_value["N/A"] += v
            continue

        prof = profiles.get(sym) or {}
        sector = prof.get("gicsSector") or prof.get("sector") or prof.get("finnhubIndustry") or "Unknown"
        country = prof.get("country") or ""
        region = _country_to_region(str(country))

        sector_value[str(sector)] += v
        region_value[region] += v

    sector_weights = {k: _pct(v, market_value) for k, v in sector_value.items()} if market_value else {}
    region_weights = {k: _pct(v, market_value) for k, v in region_value.items()} if market_value else {}

    sector_region_score = _score_sector_region(sector_weights, region_weights, insights, suggestions)

    # -----------------------
    # Quality tilt (only top N equities by weight)
    # -----------------------
    fundamentals_top_n = max(0, int(fundamentals_top_n))

    equity_ranked: List[Tuple[float, str]] = []
    for sym in equity_symbols:
        v = value_by_symbol.get(sym, 0.0)
        equity_ranked.append((v, sym))

    equity_ranked.sort(key=lambda t: t[0], reverse=True)
    top_equities = [sym for _, sym in equity_ranked[:fundamentals_top_n]] if fundamentals_top_n else []

    equity_quality_sum = 0.0
    equity_weight_sum = 0.0
    covered_weight = 0.0
    omitted_equity_weight = 0.0

    for sym in equity_symbols:
        v = value_by_symbol.get(sym, 0.0)
        w = _pct(v, market_value) if market_value else 0.0
        if sym not in top_equities:
            omitted_equity_weight += w

    for sym in top_equities:
        v = value_by_symbol.get(sym, 0.0)
        w = _pct(v, market_value) if market_value else 0.0
        if w <= 0:
            continue

        res = await fetch_fundamentals_cached(sym)
        norm = (res.data or {}).get("normalized") if res and isinstance(res.data, dict) else {}
        q = _quality_from_normalized(norm if isinstance(norm, dict) else {})

        equity_quality_sum += q * w
        equity_weight_sum += w
        if q > 0:
            covered_weight += w

    equity_quality_weighted = (equity_quality_sum / equity_weight_sum) if equity_weight_sum else 0.0
    quality_score = _score_quality(equity_quality_weighted, covered_weight, insights, suggestions)

    if omitted_equity_weight > 0.5:
        notes.append(
            f"Quality score computed on top {len(top_equities)} equities (~{(100.0 - omitted_equity_weight):.1f}% covered)."
        )
    else:
        notes.append(f"Quality score computed on top {len(top_equities)} equities.")

    # -----------------------
    # Final score
    # -----------------------
    total = div_score + risk_score + sector_region_score + quality_score
    total = max(0, min(100, int(round(total))))

    # Quick headline insight
    insights.insert(0, f"Top holding: {top1:.1f}% · Top 5: {top5:.1f}% · HHI: {hhi:.3f}")
    if crypto_w:
        insights.insert(1, f"Crypto exposure: {crypto_w:.1f}%")

    # Data-quality notes
    if sector_weights.get("Unknown", 0.0) >= 15:
        notes.append("Some equities are missing sector data; sector score is best-effort.")

    return PortfolioHealthScoreResponse(
        user_id=user_id,
        currency=curr,
        as_of=as_of,
        score=total,
        grade=_grade(total),
        baseline=baseline,
        subscores=HealthSubscores(
            diversification=div_score,
            risk_balance=risk_score,
            sector_region=sector_region_score,
            quality=quality_score,
        ),
        sector_weights_pct={
            k: round(v, 6) for k, v in sorted(sector_weights.items(), key=lambda kv: kv[1], reverse=True)
        },
        region_weights_pct={
            k: round(v, 6) for k, v in sorted(region_weights.items(), key=lambda kv: kv[1], reverse=True)
        },
        insights=insights[:12],
        suggestions=suggestions[:10],
        notes=notes,
    )
