# services/ai/risk_metrics.py
"""
Quantitative risk metrics computed from Yahoo Finance price history.
Used by both portfolio and single-stock analysis aggregators.

All price data comes from Yahoo Finance (free tier) via yahooquery.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from yahooquery import Ticker

import logging

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.045  # approximate current T-bill rate
BENCHMARK_SYMBOL = "SPY"  # S&P 500 ETF as benchmark


# ============================================================================
# CORE MATH — pure functions, no I/O
# ============================================================================

def _daily_returns(closes: pd.Series) -> pd.Series:
    """Daily percentage returns from a close price series."""
    return closes.pct_change().dropna()


def compute_volatility(returns: pd.Series) -> Optional[float]:
    """Annualized volatility from daily returns."""
    if len(returns) < 20:
        return None
    return float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(closes: pd.Series) -> Optional[float]:
    """Maximum peak-to-trough drawdown. Returns negative (e.g. -0.35 = -35%)."""
    if len(closes) < 2:
        return None
    peak = closes.cummax()
    drawdowns = (closes - peak) / peak
    return float(drawdowns.min())


def compute_sharpe(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> Optional[float]:
    """Annualized Sharpe ratio."""
    if len(returns) < 60:
        return None
    ann_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = compute_volatility(returns)
    if ann_vol is None or ann_vol < 0.001:
        return None
    return (ann_return - risk_free) / ann_vol


def compute_sortino(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> Optional[float]:
    """Annualized Sortino ratio (penalises downside volatility only)."""
    if len(returns) < 60:
        return None
    ann_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    daily_rf = risk_free / TRADING_DAYS_PER_YEAR
    downside = returns[returns < daily_rf] - daily_rf
    if len(downside) < 10:
        return None
    downside_std = float(downside.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std < 0.001:
        return None
    return (ann_return - risk_free) / downside_std


def compute_hhi(weights: List[float]) -> float:
    """
    Herfindahl-Hirschman Index from portfolio weights (0-100 scale each).
    < 1500 = diversified, 1500-2500 = moderate, > 2500 = concentrated.
    """
    return sum(w ** 2 for w in weights)


def compute_beta_from_returns(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Compute beta as cov(asset, benchmark) / var(benchmark).
    Used when Yahoo's summary_detail doesn't provide beta (e.g. crypto).
    """
    # Align both series on common dates
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 60:
        return None
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if var < 1e-10:
        return None
    return cov / var


def _round_opt(val: Optional[float], decimals: int = 4) -> Optional[float]:
    """Round a value if not None."""
    return round(val, decimals) if val is not None else None


def compute_symbol_risk(
    closes: pd.Series,
    beta: Optional[float] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute risk metrics for a single symbol given its close prices.
    
    If beta is None and benchmark_returns is provided, beta is computed
    from the covariance of asset returns against the benchmark.
    """
    returns = _daily_returns(closes)

    # If no pre-fetched beta, try computing from benchmark returns
    resolved_beta = beta
    if resolved_beta is None and benchmark_returns is not None:
        resolved_beta = compute_beta_from_returns(returns, benchmark_returns)

    return {
        "volatility_annualized": _round_opt(compute_volatility(returns)),
        "max_drawdown": _round_opt(compute_max_drawdown(closes)),
        "sharpe_ratio": _round_opt(compute_sharpe(returns)),
        "sortino_ratio": _round_opt(compute_sortino(returns)),
        "beta": _round_opt(resolved_beta, 3),
        "trading_days": len(returns),
    }


# ============================================================================
# DATA FETCHING — Yahoo Finance via yahooquery (free tier)
# ============================================================================

def _fetch_close_prices(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Batch-fetch daily close prices for one or more symbols.
    Returns a DataFrame: index=date, columns=symbol tickers.
    Uses a single HTTP call for all symbols.
    """
    if not symbols:
        return pd.DataFrame()

    syms = [s.upper() for s in symbols if s]
    tq = Ticker(" ".join(syms), asynchronous=False, formatted=False)
    df = tq.history(period=period, interval="1d")

    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()

    df = df.reset_index()

    # Prefer adjusted close when available
    price_col = "adjclose" if "adjclose" in df.columns and df["adjclose"].notna().any() else "close"
    if price_col not in df.columns:
        return pd.DataFrame()

    # Find date column
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col, price_col])

    if "symbol" in df.columns:
        # Multi-symbol (or single symbol returned with symbol column)
        df["symbol"] = df["symbol"].str.upper()
        pivot = df.pivot_table(index=date_col, columns="symbol", values=price_col)
    else:
        # Single symbol without symbol column
        df = df.set_index(date_col)[[price_col]]
        df.columns = [syms[0]]
        pivot = df

    return pivot.sort_index()


def _fetch_beta_map(symbols: List[str]) -> Dict[str, Optional[float]]:
    """
    Batch-fetch beta values from Yahoo's summaryDetail.
    Single HTTP call for all symbols.
    """
    if not symbols:
        return {}

    syms = [s.upper() for s in symbols if s]
    tq = Ticker(" ".join(syms), asynchronous=False, formatted=False)

    try:
        sd = tq.summary_detail
    except Exception:
        return {}

    result: Dict[str, Optional[float]] = {}
    if isinstance(sd, dict):
        for sym in syms:
            node = sd.get(sym, {})
            if isinstance(node, dict):
                raw_beta = node.get("beta")
                try:
                    result[sym] = float(raw_beta) if raw_beta is not None else None
                except (TypeError, ValueError):
                    result[sym] = None
            else:
                result[sym] = None

    return result


# ============================================================================
# HIGH-LEVEL API — used by aggregators
# ============================================================================

def _to_yahoo_symbol(symbol: str, asset_type: Optional[str] = None) -> str:
    """
    Convert a symbol to Yahoo Finance format.
    Crypto tickers need a '-USD' suffix (e.g. BTC → BTC-USD).
    """
    sym = symbol.strip().upper()
    if asset_type and asset_type.lower() in ("crypto", "cryptocurrency"):
        if not sym.endswith("-USD"):
            return f"{sym}-USD"
    return sym


async def fetch_symbol_risk_metrics(
    symbol: str,
    period: str = "1y",
    existing_beta: Optional[float] = None,
    asset_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute risk metrics for a single stock.
    Used by the stock analysis aggregator.

    Args:
        symbol: Ticker symbol
        period: Yahoo Finance period (default 1y)
        existing_beta: If beta is already known (from earlier Yahoo call), skip re-fetch
        asset_type: "crypto" / "cryptocurrency" / "equity" etc. — crypto symbols
                    are converted to Yahoo format (BTC → BTC-USD)
    """
    raw_sym = (symbol or "").strip().upper()
    if not raw_sym:
        return {}

    yahoo_sym = _to_yahoo_symbol(raw_sym, asset_type)

    try:
        # Fetch price history in thread pool (yahooquery is synchronous)
        closes_df = await asyncio.to_thread(_fetch_close_prices, [yahoo_sym], period)

        if closes_df.empty or yahoo_sym not in closes_df.columns:
            # Fallback: if original symbol didn't work and we haven't tried -USD, try it
            if yahoo_sym == raw_sym and not raw_sym.endswith("-USD"):
                yahoo_sym = f"{raw_sym}-USD"
                closes_df = await asyncio.to_thread(_fetch_close_prices, [yahoo_sym], period)
                if closes_df.empty or yahoo_sym not in closes_df.columns:
                    return {}
            else:
                return {}

        # Use existing beta or fetch it
        beta = existing_beta
        if beta is None:
            beta_map = await asyncio.to_thread(_fetch_beta_map, [yahoo_sym])
            beta = beta_map.get(yahoo_sym)

        return compute_symbol_risk(closes_df[yahoo_sym].dropna(), beta=beta)

    except Exception as e:
        logger.warning(f"Risk metrics failed for {raw_sym}: {e}")
        return {}


async def fetch_portfolio_risk_metrics(
    symbols: List[str],
    weights: Dict[str, float],
    period: str = "1y",
) -> Dict[str, Any]:
    """
    Compute risk metrics for a portfolio of holdings.
    Used by the portfolio analysis aggregator.

    Args:
        symbols: Top holding symbols to analyse (typically 8-10)
        weights: Dict mapping SYMBOL -> portfolio weight (0-100 scale)
        period: Yahoo Finance period for price history
    """
    if not symbols:
        logger.warning("[PortfolioRisk] called with empty symbols list")
        return {}

    clean = [s.upper() for s in symbols if s]

    # Include benchmark (SPY) in the price fetch so we can compare
    bench = BENCHMARK_SYMBOL.upper()
    price_symbols = list(dict.fromkeys(clean + [bench]))  # dedupe, preserves order

    logger.info(f"[PortfolioRisk] fetching prices for {price_symbols}, betas for {clean}")

    try:
        # Batch-fetch prices + betas in parallel (two HTTP calls total)
        closes_df, beta_map = await asyncio.gather(
            asyncio.to_thread(_fetch_close_prices, price_symbols, period),
            asyncio.to_thread(_fetch_beta_map, clean),  # beta only for holdings
        )
    except Exception as e:
        logger.warning(f"[PortfolioRisk] data fetch failed: {e}", exc_info=True)
        return {}

    if closes_df.empty:
        logger.warning(f"[PortfolioRisk] closes_df is empty for {price_symbols}")
        return {}

    logger.info(
        f"[PortfolioRisk] closes_df columns={list(closes_df.columns)}, "
        f"rows={len(closes_df)}, beta_map={beta_map}"
    )

    # ------------------------------------------------------------------
    # Per-symbol metrics
    # ------------------------------------------------------------------
    # Pre-compute benchmark returns so we can derive beta for assets
    # that don't have it from Yahoo (e.g. crypto)
    bench_returns: Optional[pd.Series] = None
    if bench in closes_df.columns:
        bench_col = closes_df[bench].dropna()
        if len(bench_col) >= 30:
            bench_returns = _daily_returns(bench_col)

    per_symbol: Dict[str, Dict[str, Any]] = {}
    available = [s for s in clean if s in closes_df.columns]
    skipped = [s for s in clean if s not in closes_df.columns]

    if skipped:
        logger.warning(f"[PortfolioRisk] symbols not found in price data: {skipped}")

    for sym in available:
        col = closes_df[sym].dropna()
        if len(col) < 30:
            logger.warning(f"[PortfolioRisk] {sym} has only {len(col)} data points, skipping")
            continue
        per_symbol[sym] = compute_symbol_risk(
            col,
            beta=beta_map.get(sym),
            benchmark_returns=bench_returns,
        )

    logger.info(f"[PortfolioRisk] computed metrics for {list(per_symbol.keys())} ({len(per_symbol)}/{len(clean)})")

    if not per_symbol:
        logger.warning("[PortfolioRisk] no per_symbol metrics computed, returning empty")
        return {}

    # ------------------------------------------------------------------
    # Portfolio-level aggregates
    # ------------------------------------------------------------------

    # Weighted beta
    total_w = 0.0
    weighted_beta = 0.0
    weighted_vol = 0.0

    for sym, m in per_symbol.items():
        w = weights.get(sym, 0) / 100.0  # % → decimal
        b = m.get("beta")
        v = m.get("volatility_annualized")
        if b is not None:
            weighted_beta += w * b
            total_w += w
        if v is not None:
            weighted_vol += w * v

    # With computed betas (from returns), most holdings should now have beta,
    # so lower the coverage threshold from 30% to 10%
    portfolio_beta = weighted_beta / total_w if total_w > 0.1 else None

    # HHI from full weight list (not just analysed symbols)
    hhi = compute_hhi(list(weights.values()))

    # Average pairwise correlation among analysed holdings
    avg_corr = None
    if len(available) >= 2:
        returns_df = closes_df[available].pct_change().dropna()
        if len(returns_df) >= 30:
            corr_matrix = returns_df.corr()
            n = len(corr_matrix)
            if n >= 2:
                total_corr = corr_matrix.values.sum() - n  # exclude diagonal (1s)
                avg_corr = total_corr / (n * (n - 1))

    # ------------------------------------------------------------------
    # Benchmark comparison (SPY)
    # ------------------------------------------------------------------
    benchmark: Dict[str, Any] = {}
    if bench in closes_df.columns:
        bench_col = closes_df[bench].dropna()
        if len(bench_col) >= 30:
            if bench_returns is None:
                bench_returns = _daily_returns(bench_col)
            bench_ann_return = float(bench_returns.mean() * TRADING_DAYS_PER_YEAR)
            benchmark = {
                "symbol": bench,
                "annualized_return": round(bench_ann_return, 4),
                "volatility": compute_volatility(bench_returns),
                "max_drawdown": compute_max_drawdown(bench_col),
                "sharpe_ratio": compute_sharpe(bench_returns),
            }
            # round where possible
            for k in ("volatility", "max_drawdown", "sharpe_ratio"):
                if benchmark.get(k) is not None:
                    benchmark[k] = round(benchmark[k], 4)

    return {
        "portfolio_beta": round(portfolio_beta, 2) if portfolio_beta is not None else None,
        "portfolio_volatility_weighted": round(weighted_vol, 4) if weighted_vol > 0 else None,
        "hhi_concentration": round(hhi, 0),
        "avg_correlation_top_holdings": round(avg_corr, 2) if avg_corr is not None else None,
        "symbols_analyzed": len(per_symbol),
        "per_symbol": per_symbol,
        "benchmark": benchmark,
    }
