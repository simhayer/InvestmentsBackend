# services/ai/technicals_pack.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from utils.common_helpers import safe_float, pct_change


Point = Dict[str, Any]
Json = Dict[str, Any]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _last_n_valid_closes(points: List[Point]) -> List[float]:
    closes: List[float] = []
    for p in points:
        c = safe_float(p.get("c"))
        if c is not None and c > 0:
            closes.append(c)
    return closes

def _last_n_valid_volumes(points: List[Point]) -> List[float]:
    vols: List[float] = []
    for p in points:
        v = safe_float(p.get("v"))
        if v is not None and v >= 0:
            vols.append(v)
    return vols

def _rolling_mean(vals: List[float], window: int) -> Optional[float]:
    if window <= 0 or len(vals) < window:
        return None
    return sum(vals[-window:]) / float(window)

def _log_returns(closes: List[float]) -> List[float]:
    rets: List[float] = []
    for i in range(1, len(closes)):
        a, b = closes[i], closes[i - 1]
        if a > 0 and b > 0:
            rets.append(math.log(a / b))
    return rets

def _realized_vol_annualized(closes: List[float], window: int = 20, trading_days: int = 252) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    r = _log_returns(closes[-(window + 1):])
    if len(r) < 2:
        return None
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var) * math.sqrt(trading_days)

def _max_drawdown(closes: List[float], window: int) -> Optional[float]:
    if window <= 1 or len(closes) < window:
        return None
    sub = closes[-window:]
    peak = sub[0]
    max_dd = 0.0
    for c in sub[1:]:
        peak = max(peak, c)
        dd = (c / peak) - 1.0  # negative or zero
        if dd < max_dd:
            max_dd = dd
    return max_dd  # e.g., -0.22 == -22%

def _swing_high_low(closes: List[float], window: int) -> Tuple[Optional[float], Optional[float]]:
    if len(closes) < window:
        return None, None
    sub = closes[-window:]
    return max(sub), min(sub)

def _series_return(closes: List[float], bars_back: int) -> Optional[float]:
    if len(closes) <= bars_back:
        return None
    return pct_change(closes[-1], closes[-1 - bars_back])

def build_technical_pack(
    symbol: str,
    points: List[Point],
    *,
    benchmark_symbol: Optional[str] = None,
    benchmark_points: Optional[List[Point]] = None,
    asof_utc: Optional[str] = None,
) -> Json:
    """
    points shape expected: [{"t": epoch, "o":..., "h":..., "l":..., "c":..., "v":...}, ...]
    Returns a compact, LLM-friendly technical pack (computed stats, not raw candles).
    """
    sym = (symbol or "").upper().strip()
    asof = asof_utc or _now_utc_iso()

    if not points:
        return {"status": "ok", "symbol": sym, "asof_utc": asof, "error": None, "is_empty": True}

    # Sort by time just in case
    pts = sorted(points, key=lambda x: int(x.get("t") or 0))

    closes = _last_n_valid_closes(pts)
    vols = _last_n_valid_volumes(pts)

    if len(closes) < 30:
        # Not enough history to do most technical stats credibly
        last_close = closes[-1] if closes else None
        return {
            "status": "ok",
            "symbol": sym,
            "asof_utc": asof,
            "is_empty": False,
            "data_quality": {"warning": "Insufficient price history for full technicals", "closes": len(closes)},
            "last_close": last_close,
        }

    last_close = closes[-1]

    # Common horizons in trading days (approx)
    horizons = {
        "1d": 1,
        "5d": 5,
        "1m": 21,
        "3m": 63,
        "6m": 126,
        "1y": 252,
    }

    returns = {k: _series_return(closes, bars) for k, bars in horizons.items()}

    ma20 = _rolling_mean(closes, 20)
    ma50 = _rolling_mean(closes, 50)
    ma200 = _rolling_mean(closes, 200)

    trend = {
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "above_ma20": (last_close > ma20) if (last_close is not None and ma20 is not None) else None,
        "above_ma50": (last_close > ma50) if (last_close is not None and ma50 is not None) else None,
        "above_ma200": (last_close > ma200) if (last_close is not None and ma200 is not None) else None,
    }

    vol20 = _realized_vol_annualized(closes, window=20)
    vol60 = _realized_vol_annualized(closes, window=60) if len(closes) >= 61 else None

    dd_6m = _max_drawdown(closes, window=min(126, len(closes)))
    dd_1y = _max_drawdown(closes, window=min(252, len(closes)))

    # Simple swing levels (not “support/resistance”, just recent range)
    hi_20, lo_20 = _swing_high_low(closes, 20)
    hi_60, lo_60 = _swing_high_low(closes, 60)

    # Liquidity proxy: avg dollar volume over 20d/60d
    # Use aligned closes + vols if possible; otherwise approximate with last_close * avg(volume)
    avg_vol_20 = None
    avg_vol_60 = None
    if len(vols) >= 60:
        avg_vol_20 = sum(vols[-20:]) / 20.0
        avg_vol_60 = sum(vols[-60:]) / 60.0
    elif len(vols) >= 20:
        avg_vol_20 = sum(vols[-20:]) / 20.0

    avg_dollar_vol_20 = (avg_vol_20 * last_close) if (avg_vol_20 is not None and last_close is not None) else None
    avg_dollar_vol_60 = (avg_vol_60 * last_close) if (avg_vol_60 is not None and last_close is not None) else None

    relative: Optional[Json] = None
    if benchmark_symbol and benchmark_points:
        bench_closes = _last_n_valid_closes(sorted(benchmark_points, key=lambda x: int(x.get("t") or 0)))
        # Compare only if benchmark has enough history too
        if len(bench_closes) >= 30 and len(closes) >= 30:
            bench_returns = {k: _series_return(bench_closes, bars) for k, bars in horizons.items()}
            # Outperformance = stock return - benchmark return
            rel = {}
            for k in horizons.keys():
                a = returns.get(k)
                b = bench_returns.get(k)
                rel[k] = (a - b) if (a is not None and b is not None) else None

            relative = {
                "benchmark": benchmark_symbol.upper().strip(),
                "return_vs_benchmark_pct": rel,
                "benchmark_returns": bench_returns,
            }

    return {
        "status": "ok",
        "symbol": sym,
        "asof_utc": asof,
        "is_empty": False,
        "last_close": last_close,
        "returns_pct": returns,
        "trend": trend,
        "volatility": {
            "realized_20d_ann": vol20,
            "realized_60d_ann": vol60,
        },
        "drawdown": {
            "max_6m": dd_6m,  # negative numbers
            "max_1y": dd_1y,
        },
        "ranges": {
            "swing_20d": {"high": hi_20, "low": lo_20},
            "swing_60d": {"high": hi_60, "low": lo_60},
        },
        "liquidity": {
            "avg_vol_20d": avg_vol_20,
            "avg_vol_60d": avg_vol_60,
            "avg_dollar_vol_20d": avg_dollar_vol_20,
            "avg_dollar_vol_60d": avg_dollar_vol_60,
        },
        "relative": relative,
        "data_quality": {
            "points": len(pts),
            "closes": len(closes),
            "volumes": len(vols),
        },
    }

def _r(x: Any, nd: int) -> Optional[float]:
    return round(x, nd) if isinstance(x, (int, float)) else None

def compact_tech_pack(tp: Json) -> Json:
    """
    Reduce float noise, add convenience flags, and make the pack LLM-friendly.
    Does NOT change meaning, only presentation.
    """
    if not isinstance(tp, dict):
        return tp

    out = dict(tp)

    # --- last price ---
    out["last_close"] = _r(out.get("last_close"), 2)

    # --- returns ---
    rp = out.get("returns_pct") or {}
    out["returns_pct"] = {k: _r(v, 2) for k, v in rp.items()}

    # --- trend / MAs ---
    tr = out.get("trend") or {}
    ma20 = tr.get("ma20")
    ma50 = tr.get("ma50")
    ma200 = tr.get("ma200")

    out["trend"] = {
        "ma20": _r(ma20, 2),
        "ma50": _r(ma50, 2),
        "ma200": _r(ma200, 2),
        "above_ma20": tr.get("above_ma20"),
        "above_ma50": tr.get("above_ma50"),
        "above_ma200": tr.get("above_ma200"),
    }

    # Trend regime (simple, explainable)
    if isinstance(ma20, (int, float)) and isinstance(ma50, (int, float)) and isinstance(ma200, (int, float)):
        if ma20 > ma50 > ma200:
            trend_regime = "uptrend"
        elif ma20 < ma50 < ma200:
            trend_regime = "downtrend"
        else:
            trend_regime = "mixed"
    else:
        trend_regime = None

    out["trend_regime"] = trend_regime

    # --- volatility ---
    vol = out.get("volatility") or {}
    out["volatility"] = {
        "realized_20d_ann": _r(vol.get("realized_20d_ann"), 3),
        "realized_60d_ann": _r(vol.get("realized_60d_ann"), 3),
    }

    # --- drawdown ---
    dd = out.get("drawdown") or {}
    out["drawdown"] = {
        "max_6m": _r(dd.get("max_6m"), 3),
        "max_1y": _r(dd.get("max_1y"), 3),
    }

    # --- ranges ---
    ranges = out.get("ranges") or {}
    s20 = ranges.get("swing_20d") or {}
    hi20 = s20.get("high")
    lo20 = s20.get("low")
    last = out.get("last_close")

    range_pos_20d = None
    if isinstance(last, (int, float)) and isinstance(hi20, (int, float)) and isinstance(lo20, (int, float)) and hi20 != lo20:
        range_pos_20d = round((last - lo20) / (hi20 - lo20), 2)

    out["ranges"] = {
        "swing_20d": {"high": _r(hi20, 2), "low": _r(lo20, 2)},
        "swing_60d": {
            "high": _r((ranges.get("swing_60d") or {}).get("high"), 2),
            "low": _r((ranges.get("swing_60d") or {}).get("low"), 2),
        },
        "range_position_20d": range_pos_20d,  # 0 = near low, 1 = near high
    }

    # --- liquidity ---
    liq = out.get("liquidity") or {}
    out["liquidity"] = {
        "avg_vol_20d": _r(liq.get("avg_vol_20d"), 0),
        "avg_vol_60d": _r(liq.get("avg_vol_60d"), 0),
        "avg_dollar_vol_20d": _r(liq.get("avg_dollar_vol_20d"), 0),
        "avg_dollar_vol_60d": _r(liq.get("avg_dollar_vol_60d"), 0),
    }

    # --- relative vs benchmark ---
    rel = out.get("relative")
    if isinstance(rel, dict):
        rvb = rel.get("return_vs_benchmark_pct") or {}
        br = rel.get("benchmark_returns") or {}

        out["relative"] = {
            "benchmark": rel.get("benchmark"),
            "return_vs_benchmark_pct": {k: _r(v, 2) for k, v in rvb.items()},
            "benchmark_returns": {k: _r(v, 2) for k, v in br.items()},
        }

    return out