# analyze_crypto_aggregator.py
"""
Aggregates crypto asset data from Yahoo Finance (free) for AI analysis.

Unlike stocks, crypto has no fundamentals (P/E, margins, earnings).
We focus on: price action, market position, risk metrics, technical context, and news.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CryptoDataBundle:
    """All data needed for AI analysis of a single crypto asset."""
    symbol: str  # Original symbol (e.g. "ETH")
    yahoo_symbol: str = ""  # Yahoo format (e.g. "ETH-USD")

    # Price data (from Yahoo)
    name: str = ""
    currency: str = "USD"
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    day_change_pct: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    circulating_supply: Optional[float] = None

    # 52-week range
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    # Moving averages
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None

    # News
    news: List[Dict[str, Any]] = field(default_factory=list)

    # Quantitative risk metrics (from Yahoo price history via risk_metrics.py)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)

    # Data quality
    gaps: List[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "yahooSymbol": self.yahoo_symbol,
            "name": self.name,
            "currency": self.currency,
            "currentPrice": self.current_price,
            "previousClose": self.previous_close,
            "dayChangePct": self.day_change_pct,
            "marketCap": self.market_cap,
            "volume24h": self.volume_24h,
            "circulatingSupply": self.circulating_supply,
            "high52w": self.high_52w,
            "low52w": self.low_52w,
            "ma50": self.ma_50,
            "ma200": self.ma_200,
            "news": self.news,
            "riskMetrics": self.risk_metrics,
            "gaps": self.gaps,
            "fetchedAt": self.fetched_at,
        }

    def to_ai_context(self, max_news: int = 5) -> str:
        """Format crypto data as structured context for LLM analysis."""
        lines = [f"# Crypto Asset Analysis: {self.symbol}", ""]

        # ── Market Overview ──────────────────────────────────────────
        lines.append("## Market Overview")
        lines.append(f"- Asset: {self.name or self.symbol}")

        if self.current_price is not None:
            lines.append(f"- Current Price: ${self.current_price:,.4f}" if self.current_price < 1 else f"- Current Price: ${self.current_price:,.2f}")

        if self.day_change_pct is not None:
            lines.append(f"- 24h Change: {self.day_change_pct:+.2f}%")

        if self.market_cap is not None:
            if self.market_cap >= 1e9:
                lines.append(f"- Market Cap: ${self.market_cap / 1e9:,.1f}B")
            else:
                lines.append(f"- Market Cap: ${self.market_cap / 1e6:,.1f}M")

        if self.volume_24h is not None:
            if self.volume_24h >= 1e9:
                lines.append(f"- 24h Volume: ${self.volume_24h / 1e9:,.1f}B")
            elif self.volume_24h >= 1e6:
                lines.append(f"- 24h Volume: ${self.volume_24h / 1e6:,.1f}M")

        if self.circulating_supply is not None:
            lines.append(f"- Circulating Supply: {self.circulating_supply:,.0f}")

        lines.append("")

        # ── Technical Context (52-week, MAs) ─────────────────────────
        if self.high_52w and self.low_52w and self.current_price:
            pct_from_high = ((self.high_52w - self.current_price) / self.high_52w) * 100
            pct_from_low = ((self.current_price - self.low_52w) / self.low_52w) * 100 if self.low_52w else 0
            range_pct = ((self.current_price - self.low_52w) / (self.high_52w - self.low_52w)) * 100 if self.high_52w != self.low_52w else 50

            lines.append("## Technical Context")
            lines.append(f"- 52-Week Range: ${self.low_52w:,.2f} — ${self.high_52w:,.2f}")
            lines.append(f"- Distance from 52W High: -{pct_from_high:.1f}%")
            lines.append(f"- Distance from 52W Low: +{pct_from_low:.1f}%")
            lines.append(f"- Range Position: {range_pct:.0f}% (0%=low, 100%=high)")

            if self.ma_50 and self.current_price:
                direction = "above" if self.current_price > self.ma_50 else "below"
                lines.append(f"- Trading {direction} 50-day MA (${self.ma_50:,.2f})")
            if self.ma_200 and self.current_price:
                direction = "above" if self.current_price > self.ma_200 else "below"
                lines.append(f"- Trading {direction} 200-day MA (${self.ma_200:,.2f})")

            # Golden/death cross
            if self.ma_50 and self.ma_200:
                if self.ma_50 > self.ma_200:
                    lines.append("- 50-day MA above 200-day MA (bullish golden cross structure)")
                else:
                    lines.append("- 50-day MA below 200-day MA (bearish death cross structure)")

            lines.append("")

        # ── Quantitative Risk Metrics ────────────────────────────────
        rm = self.risk_metrics
        if rm:
            lines.append("## Risk Metrics (Trailing 1Y)")
            beta = rm.get("beta")
            vol = rm.get("volatility_annualized")
            mdd = rm.get("max_drawdown")
            sharpe = rm.get("sharpe_ratio")
            sortino = rm.get("sortino_ratio")
            days = rm.get("trading_days")

            if beta is not None:
                label = "low correlation to equities" if abs(beta) < 0.5 else "moderate equity correlation" if abs(beta) < 1.0 else "high equity correlation"
                lines.append(f"- Beta vs S&P 500: {beta:.2f} ({label})")
            if vol is not None:
                vol_label = "moderate" if vol < 0.5 else "high" if vol < 0.8 else "extreme"
                lines.append(f"- Annualized Volatility: {vol * 100:.1f}% ({vol_label})")
            if mdd is not None:
                lines.append(f"- Max Drawdown (1Y): {mdd * 100:.1f}%")
            if sharpe is not None:
                sharpe_label = "deeply negative" if sharpe < -0.5 else "negative" if sharpe < 0 else "poor" if sharpe < 0.5 else "adequate" if sharpe < 1.0 else "good"
                lines.append(f"- Sharpe Ratio: {sharpe:.2f} ({sharpe_label})")
            if sortino is not None:
                lines.append(f"- Sortino Ratio: {sortino:.2f}")
            if days is not None:
                lines.append(f"- Based on {days} trading days")
            lines.append("")

        # ── Recent News ──────────────────────────────────────────────
        if self.news:
            lines.append("## Recent News")
            for n in self.news[:max_news]:
                title = n.get("title", "")[:100]
                date = (n.get("published_at") or "")[:10]
                lines.append(f"- [{date}] {title}")
            lines.append("")

        # ── Data Gaps ────────────────────────────────────────────────
        if self.gaps:
            lines.append("## Data Limitations")
            lines.append("- This is a cryptocurrency — no P/E, margins, earnings, or analyst targets are available.")
            for gap in self.gaps:
                lines.append(f"- {gap}")

        return "\n".join(lines)


# ============================================================================
# AGGREGATION PIPELINE
# ============================================================================

def _to_yahoo_crypto(symbol: str) -> str:
    """Convert plain crypto ticker to Yahoo Finance format."""
    sym = symbol.strip().upper()
    if not sym.endswith("-USD"):
        return f"{sym}-USD"
    return sym


async def aggregate_crypto_data(
    symbol: str,
    *,
    include_news: bool = True,
    news_days_back: int = 7,
    news_limit: int = 6,
) -> CryptoDataBundle:
    """
    Aggregates all available data for a crypto asset.

    Data sources (all free):
    - Yahoo Finance (via yahooquery): price, market cap, volume, 52w range, MAs
    - Yahoo price history: for risk metrics (vol, Sharpe, Sortino, max DD, beta vs SPY)
    - Finnhub news: for major crypto tickers
    """
    from yahooquery import Ticker
    from services.yahoo_service import get_full_stock_data

    sym = (symbol or "").strip().upper()
    if not sym:
        return CryptoDataBundle(symbol="", gaps=["Missing symbol"])

    yahoo_sym = _to_yahoo_crypto(sym)
    bundle = CryptoDataBundle(symbol=sym, yahoo_symbol=yahoo_sym)

    # ── 1. Fetch Yahoo market data ───────────────────────────────
    try:
        yahoo_result = await asyncio.to_thread(get_full_stock_data, yahoo_sym)
        if isinstance(yahoo_result, dict) and yahoo_result.get("status") != "error":
            bundle.name = yahoo_result.get("name") or sym
            bundle.currency = yahoo_result.get("currency") or "USD"
            bundle.current_price = yahoo_result.get("current_price")
            bundle.previous_close = yahoo_result.get("previous_close")
            bundle.market_cap = yahoo_result.get("market_cap")

            # Day change
            cp = bundle.current_price
            pc = bundle.previous_close
            if cp is not None and pc is not None and pc != 0:
                bundle.day_change_pct = ((cp - pc) / pc) * 100

            # 52-week range
            bundle.high_52w = yahoo_result.get("52_week_high")
            bundle.low_52w = yahoo_result.get("52_week_low")
        else:
            # Fallback: try fetching basic price modules directly
            logger.info(f"[Crypto Aggregator] get_full_stock_data failed for {yahoo_sym}, trying modules")
            try:
                tq = Ticker(yahoo_sym, asynchronous=False, formatted=False)
                modules = tq.get_modules(["price", "summaryDetail"])
                node = modules.get(yahoo_sym, modules) if isinstance(modules, dict) else {}

                price = node.get("price", {})
                sd = node.get("summaryDetail", {})

                bundle.name = price.get("shortName") or price.get("longName") or sym
                bundle.currency = price.get("currency") or "USD"
                bundle.current_price = price.get("regularMarketPrice")
                bundle.previous_close = price.get("regularMarketPreviousClose") or sd.get("previousClose")
                bundle.market_cap = price.get("marketCap") or sd.get("marketCap")
                bundle.volume_24h = price.get("regularMarketVolume") or sd.get("volume")
                bundle.high_52w = sd.get("fiftyTwoWeekHigh")
                bundle.low_52w = sd.get("fiftyTwoWeekLow")
                bundle.ma_50 = sd.get("fiftyDayAverage")
                bundle.ma_200 = sd.get("twoHundredDayAverage")
                bundle.circulating_supply = sd.get("circulatingSupply")
            except Exception as e2:
                bundle.gaps.append(f"Yahoo modules fetch failed: {e2}")
    except Exception as e:
        bundle.gaps.append(f"Yahoo data fetch failed: {e}")

    # Fill MAs from summaryDetail if not yet populated
    if bundle.ma_50 is None or bundle.ma_200 is None:
        try:
            tq = Ticker(yahoo_sym, asynchronous=False, formatted=False)
            sd = tq.summary_detail
            if isinstance(sd, dict):
                node = sd.get(yahoo_sym, sd)
                if isinstance(node, dict):
                    if bundle.ma_50 is None:
                        bundle.ma_50 = node.get("fiftyDayAverage")
                    if bundle.ma_200 is None:
                        bundle.ma_200 = node.get("twoHundredDayAverage")
                    if bundle.high_52w is None:
                        bundle.high_52w = node.get("fiftyTwoWeekHigh")
                    if bundle.low_52w is None:
                        bundle.low_52w = node.get("fiftyTwoWeekLow")
                    if bundle.volume_24h is None:
                        bundle.volume_24h = node.get("volume")
                    if bundle.circulating_supply is None:
                        bundle.circulating_supply = node.get("circulatingSupply")
        except Exception:
            pass  # MAs are nice-to-have, not critical

    # ── 2. Fetch risk metrics (price history + beta vs SPY) ──────
    try:
        from services.ai.risk_metrics import fetch_symbol_risk_metrics

        bundle.risk_metrics = await fetch_symbol_risk_metrics(
            yahoo_sym, period="1y", asset_type="crypto"
        )
    except Exception as e:
        bundle.gaps.append(f"Risk metrics computation failed: {e}")

    # ── 3. Fetch news ────────────────────────────────────────────
    if include_news:
        try:
            from services.finnhub.finnhub_news_service import get_company_news_cached

            # Finnhub uses the plain symbol for crypto news (e.g., "BINANCE:ETHUSDT")
            # but also indexes general crypto news under the plain ticker
            news_result = await get_company_news_cached(
                sym, days_back=news_days_back, limit=news_limit
            )
            if isinstance(news_result, dict):
                bundle.news = news_result.get("items", [])
            elif isinstance(news_result, list):
                bundle.news = news_result
        except Exception as e:
            bundle.gaps.append(f"News fetch failed: {e}")

    if bundle.current_price is None:
        bundle.gaps.append("Could not fetch current price — data may be limited")

    return bundle
