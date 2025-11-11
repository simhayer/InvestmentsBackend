from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import yfinance as yf


# ----------------------------- Types -----------------------------

@dataclass
class PlaidPosition:
    symbol: str                 # e.g., "AAPL" or "VFV.TO"
    quantity: float
    cost_basis: float           # total cost in base_currency (your accounting), or None -> we estimate
    name: Optional[str] = None
    asset_class: Optional[str] = None  # "equity" | "etf" | "fund" | "crypto" | "bond" | "cash" | "other"
    sector: Optional[str] = None
    region: Optional[str] = None
    # If Plaid provides per-lot currency/cost, you can add fields here.


# --------------------------- FX Helpers --------------------------

def _fx_pair(base: str, quote: str) -> str:
    """
    Yahoo FX tickers look like 'USDCAD=X' meaning 1 USD in CAD.
    We return '<base><quote>=X' meaning 1 BASE in QUOTE.
    """
    return f"{base}{quote}=X"


def get_fx_rate(base: str, quote: str) -> float:
    """
    Return rate to convert 1 unit of `base` into `quote`.
    If base == quote, returns 1.0.
    """
    if base == quote:
        return 1.0
    pair = _fx_pair(base, quote)
    t = yf.Ticker(pair)
    px = None
    # Prefer fast info, then price in history
    try:
        info = t.fast_info
        px = getattr(info, "last_price", None)
    except Exception:
        px = None
    if px is None:
        hist = t.history(period="5d", interval="1d")
        if not hist.empty:
            px = float(hist["Close"].dropna().iloc[-1])
    if px is None or not math.isfinite(px):
        # Fallback via inverse pair (e.g., CADUSD if USDCAD missing)
        inv_pair = _fx_pair(quote, base)
        t2 = yf.Ticker(inv_pair)
        hist2 = t2.history(period="5d", interval="1d")
        if not hist2.empty:
            inv = float(hist2["Close"].dropna().iloc[-1])
            if inv and inv != 0:
                return 1.0 / inv
        raise RuntimeError(f"Could not fetch FX rate {pair} or inverse {inv_pair}")
    return float(px)


# -------------------------- Metric Utils -------------------------

def pct_return(series: pd.Series, lookback_days: int) -> Optional[float]:
    """Percent return over last N trading days (approx)."""
    if series is None or series.size < lookback_days + 1:
        return None
    start = float(series.dropna().iloc[-(lookback_days + 1)])
    end = float(series.dropna().iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0


def rolling_max_drawdown(close: pd.Series) -> Optional[float]:
    """Max drawdown (%) over the available window (expects daily closes)."""
    if close is None or close.size < 10:
        return None
    cummax = close.cummax()
    dd = (close / cummax - 1.0) * 100.0
    return float(dd.min())  # negative percentage


def daily_vol_pct(close: pd.Series, window: int = 30) -> Optional[float]:
    """Non-annualized stdev of daily returns over last `window` days, in %."""
    if close is None or close.size < window + 1:
        return None
    rets = close.pct_change().dropna().iloc[-window:]
    if rets.empty:
        return None
    return float(rets.std() * 100.0)


def beta_vs_benchmark(asset_close: pd.Series, bench_close: pd.Series) -> Optional[float]:
    """Simple beta using daily returns over the overlapping window."""
    if asset_close is None or bench_close is None:
        return None
    df = pd.concat(
        [asset_close.rename("a"), bench_close.rename("b")], axis=1
    ).dropna()
    if df.shape[0] < 30:
        return None
    r = df.pct_change().dropna()
    if r.empty or r["b"].var() == 0:
        return None
    cov = np.cov(r["a"], r["b"])[0, 1]
    beta = cov / r["b"].var()
    return float(beta)


# ------------------------ Core Fetch Function --------------------

def build_metrics_from_plaid(
    plaid_positions: List[Dict[str, Any]],
    base_currency: str = "CAD",
    benchmark_ticker: str = "SPY",
    classification_map: Optional[Dict[str, str]] = None,  # e.g., {"AAPL":"core","MNMD":"speculative"}
) -> Dict[str, Any]:
    """
    Given Plaid positions and a base currency, pull Yahoo data and compute metrics JSON.

    plaid_positions: list of dicts with keys:
        - symbol (str)               REQUIRED
        - quantity (float)           REQUIRED
        - cost_basis (float)         REQUIRED (total cost in base_currency; if None, returns_pct work but PnL may be None)
        - name (str)                 optional
        - asset_class (str)          optional ("equity","etf","fund","crypto","bond","cash","other")
        - sector (str)               optional
        - region (str)               optional

    Returns METRICS_SCHEMA-compliant dict:
        {
          "per_symbol": { SYMBOL: {...}, ... },
          "portfolio": {...}
        }
    """
    # Normalize Plaid input
    positions: List[PlaidPosition] = []
    for p in plaid_positions:
        positions.append(
            PlaidPosition(
                symbol=p["symbol"].strip().upper(),
                quantity=float(p.get("quantity", 0.0) or 0.0),
                cost_basis=float(p.get("cost_basis", 0.0) or 0.0),
                name=p.get("name"),
                asset_class=p.get("asset_class"),
                sector=p.get("sector"),
                region=p.get("region"),
            )
        )

    if not positions:
        return {"per_symbol": {}, "portfolio": {
            "total_value": 0.0, "cash_value": 0.0, "num_positions": 0,
            "concentration_top_5_pct": 0.0,
            "core_weight_pct": 0.0, "speculative_weight_pct": 0.0, "hedge_weight_pct": 0.0
        }}

    # Pull all tickers in one go where possible
    tickers = sorted({p.symbol for p in positions})
    y_objs = {t: yf.Ticker(t) for t in tickers}

    # Benchmark price history (for beta)
    bench_close = None
    try:
        bench_hist = yf.Ticker(benchmark_ticker).history(period="1y", interval="1d")["Close"]
        bench_close = bench_hist.dropna() if not bench_hist.empty else None
    except Exception:
        bench_close = None

    per_symbol: Dict[str, Any] = {}
    total_value = 0.0
    cash_value = 0.0

    # FX cache: currency -> rate to base_currency
    fx_cache: Dict[Tuple[str, str], float] = {}

    for p in positions:
        t = y_objs[p.symbol]

        # ---------- Current price & currency ----------
        try:
            info = t.fast_info
        except Exception:
            info = None

        # Currency reported by Yahoo (best-effort)
        y_curr = None
        last_px = None
        if info is not None:
            y_curr = getattr(info, "currency", None)
            last_px = getattr(info, "last_price", None)

        # Fallbacks
        if last_px is None:
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                last_px = float(hist["Close"].dropna().iloc[-1])

        # Try to pick up currency if missing
        if y_curr is None:
            try:
                y_curr = t.info.get("currency")
            except Exception:
                y_curr = None

        # Default unknown currency to base (safer for math), though we try hard to get it
        if y_curr is None:
            y_curr = base_currency

        if last_px is None or not math.isfinite(last_px):
            # If we can't price it, treat as zero-value (e.g., delisted); still include structure
            last_px = 0.0

        # ---------- FX conversion to base_currency ----------
        key = (y_curr, base_currency)
        if key not in fx_cache:
            fx_cache[key] = get_fx_rate(y_curr, base_currency)
        fx_rate = fx_cache[key]  # 1 y_curr -> ? base_currency

        px_base = float(last_px) * fx_rate
        mv = px_base * float(p.quantity)  # market value in base

        total_value += mv
        if (p.asset_class or "").lower() == "cash" or p.symbol in ("CASH",):
            cash_value += mv

        # ---------- History for returns/vol/DD/beta ----------
        # 1y of daily prices (close)
        hist_1y = t.history(period="1y", interval="1d")
        close_1y = hist_1y["Close"].dropna() if not hist_1y.empty else pd.Series(dtype=float)

        # Compute returns in instrument currency (price-based), unaffected by quantity
        r_1d = None
        r_1w = None
        if not close_1y.empty:
            # 1D: last vs prev close
            if close_1y.size >= 2:
                r_1d = float((close_1y.iloc[-1] / close_1y.iloc[-2] - 1.0) * 100.0)
            # 1W ~ 5 trading days
            r_1w = pct_return(close_1y, 5)
            r_1m = pct_return(close_1y, 21)
            r_3m = pct_return(close_1y, 63)
            r_1y = pct_return(close_1y, min(252, close_1y.size - 1))
            vol30 = daily_vol_pct(close_1y, 30)
            mdd1y = rolling_max_drawdown(close_1y)
            beta1y = beta_vs_benchmark(close_1y, bench_close) if bench_close is not None else None
        else:
            r_1m = r_3m = r_1y = vol30 = mdd1y = beta1y = None

        # ---------- PnL based on provided cost_basis ----------
        cb = float(p.cost_basis or 0.0)
        if cb > 0:
            unreal_abs = mv - cb
            unreal_pct = (unreal_abs / cb) * 100.0
        else:
            unreal_abs = None
            unreal_pct = None

        # ---------- Build per_symbol record ----------
        rec = {
            "symbol": p.symbol,
            "name": p.name or p.symbol,
            "asset_class": (p.asset_class or "other").lower(),
            "sector": p.sector or "",
            "region": p.region or "",
            "weight_pct": 0.0,  # fill later after total_value known
            "market_value": round(mv, 2),
            "cost_basis": round(cb, 2) if cb is not None else 0.0,
            "unrealized_pnl_abs": round(unreal_abs, 2) if unreal_abs is not None else None,
            "unrealized_pnl_pct": round(unreal_pct, 2) if unreal_pct is not None else None,
            "return_1D_pct": None if r_1d is None else round(r_1d, 2),
            "return_1W_pct": None if r_1w is None else round(r_1w, 2),
            "return_1M_pct": None if r_1m is None else round(r_1m, 2),
            "return_3M_pct": None if r_3m is None else round(r_3m, 2),
            "return_1Y_pct": None if r_1y is None else round(r_1y, 2),
            "vol_30D_pct": None if vol30 is None else round(vol30, 2),
            "max_drawdown_1Y_pct": None if mdd1y is None else round(mdd1y, 2),
            "beta_1Y": None if beta1y is None else round(beta1y, 2),
            "is_leveraged": bool("lever" in (p.name or "").lower() or p.symbol.endswith(("UP", "DOWN", "BULL", "BEAR"))),
            # You could refine leveraged detection by parsing t.info if needed.
        }
        per_symbol[p.symbol] = rec

    # ---------- Fill weights now that total_value known ----------
    if total_value <= 0:
        for s in per_symbol.values():
            s["weight_pct"] = 0.0
    else:
        for s in per_symbol.values():
            s["weight_pct"] = round(100.0 * s["market_value"] / total_value, 2)

    # ---------- Concentration (top 5) ----------
    weights_sorted = sorted([s["weight_pct"] for s in per_symbol.values()], reverse=True)
    concentration_top_5_pct = round(sum(weights_sorted[:5]), 2) if weights_sorted else 0.0

    # ---------- Aggregations ----------
    sector_weights: Dict[str, float] = {}
    asset_class_weights: Dict[str, float] = {}
    region_weights: Dict[str, float] = {}
    for s in per_symbol.values():
        sector = s.get("sector") or "Unassigned"
        asset_class = s.get("asset_class") or "other"
        region = s.get("region") or "Unassigned"
        w = s["weight_pct"]
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
        asset_class_weights[asset_class] = asset_class_weights.get(asset_class, 0.0) + w
        region_weights[region] = region_weights.get(region, 0.0) + w

    # ---------- Portfolio risk (optional) ----------
    # Crude estimate using weighted average of symbol vol (not mathematically perfect)
    vols = [s["vol_30D_pct"] for s in per_symbol.values() if s["vol_30D_pct"] is not None]
    weights = [s["weight_pct"] for s in per_symbol.values() if s["vol_30D_pct"] is not None]
    port_vol_30d = None
    if vols and sum(weights) > 0:
        # Weighted average of non-annualized vols (proxy)
        port_vol_30d = round(sum(v * w for v, w in zip(vols, weights)) / sum(weights), 2)

    # Approx portfolio max drawdown as weighted avg of components' mdd (very rough)
    mdds = [s["max_drawdown_1Y_pct"] for s in per_symbol.values() if s["max_drawdown_1Y_pct"] is not None]
    weights_mdd = [s["weight_pct"] for s in per_symbol.values() if s["max_drawdown_1Y_pct"] is not None]
    port_mdd_1y = None
    if mdds and sum(weights_mdd) > 0:
        port_mdd_1y = round(sum(dd * w for dd, w in zip(mdds, weights_mdd)) / sum(weights_mdd), 2)

    # ---------- Core/Spec/Hedge from classification_map ----------
    core_w = spec_w = hedge_w = 0.0
    if classification_map:
        for sym, s in per_symbol.items():
            cat = (classification_map.get(sym) or "").lower()
            if cat == "core":
                core_w += s["weight_pct"]
            elif cat == "speculative":
                spec_w += s["weight_pct"]
            elif cat == "hedge":
                hedge_w += s["weight_pct"]
        core_w = round(core_w, 2)
        spec_w = round(spec_w, 2)
        hedge_w = round(hedge_w, 2)

    portfolio = {
        "total_value": round(total_value, 2),
        "cash_value": round(cash_value, 2),
        "num_positions": len(per_symbol),
        "concentration_top_5_pct": concentration_top_5_pct,
        "core_weight_pct": core_w,
        "speculative_weight_pct": spec_w,
        "hedge_weight_pct": hedge_w,
        "sector_weights_pct": {k: round(v, 2) for k, v in sector_weights.items()},
        "asset_class_weights_pct": {k: round(v, 2) for k, v in asset_class_weights.items()},
        "region_weights_pct": {k: round(v, 2) for k, v in region_weights.items()},
        "vol_30D_pct": port_vol_30d,
        "max_drawdown_1Y_pct": port_mdd_1y,
    }

    return {"per_symbol": per_symbol, "portfolio": portfolio}
