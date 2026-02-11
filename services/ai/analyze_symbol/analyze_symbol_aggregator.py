# stock_analysis_aggregator.py
"""
Aggregates all stock data from Finnhub + Yahoo for AI analysis.
Single entry point that gathers fundamentals, analyst data, news, and chart context.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class StockDataBundle:
    """All data needed for AI analysis of a single stock."""
    symbol: str
    
    # Core data
    profile: Dict[str, Any] = field(default_factory=dict)
    quote: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    normalized: Dict[str, Any] = field(default_factory=dict)
    
    # Earnings history
    earnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analyst data
    analyst_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    price_target: Optional[Dict[str, Any]] = None
    
    # News
    news: List[Dict[str, Any]] = field(default_factory=list)
    
    # Peers (for context)
    peers: List[str] = field(default_factory=list)
    
    # Yahoo supplemental data
    yahoo_data: Dict[str, Any] = field(default_factory=dict)
    
    # Data quality tracking
    gaps: List[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "profile": self.profile,
            "quote": self.quote,
            "metrics": self.metrics,
            "normalized": self.normalized,
            "earnings": self.earnings,
            "analystRecommendations": self.analyst_recommendations,
            "priceTarget": self.price_target,
            "news": self.news,
            "peers": self.peers,
            "yahooData": self.yahoo_data,
            "gaps": self.gaps,
            "fetchedAt": self.fetched_at,
        }

    def to_ai_context(self, max_news: int = 5) -> str:
        """
        Formats the data bundle as a structured context string for LLM prompts.
        """
        lines = [f"# Stock Analysis Data: {self.symbol}", ""]
        
        # Company overview
        if self.profile:
            lines.append("## Company Profile")
            lines.append(f"- Name: {self.profile.get('name', 'N/A')}")
            lines.append(f"- Industry: {self.profile.get('finnhubIndustry', 'N/A')}")
            lines.append(f"- Market Cap: ${self._fmt_num(self.normalized.get('market_cap'))}M")
            lines.append("")
        
        # Current price
        if self.quote:
            c = self.quote.get('c') or self.quote.get('currentPrice')
            pc = self.quote.get('pc') or self.quote.get('previousClose')
            if c and pc:
                change_pct = ((c - pc) / pc) * 100 if pc else 0
                lines.append("## Current Price")
                lines.append(f"- Price: ${c:.2f}")
                lines.append(f"- Change: {change_pct:+.2f}%")
                lines.append("")
        
        # Valuation metrics
        lines.append("## Valuation")
        lines.append(f"- P/E (TTM): {self._fmt_num(self.normalized.get('pe_ttm'))}")
        
        # Add forward PE from Yahoo if available
        fwd_pe = self.yahoo_data.get('defaultKeyStatistics', {}).get('forwardPE')
        if fwd_pe:
            lines.append(f"- P/E (Forward): {self._fmt_num(fwd_pe)}")
        
        peg = self.yahoo_data.get('defaultKeyStatistics', {}).get('pegRatio')
        if peg:
            lines.append(f"- PEG Ratio: {self._fmt_num(peg)}")
        lines.append("")
        
        # Profitability
        lines.append("## Profitability")
        lines.append(f"- Gross Margin: {self._fmt_pct(self.normalized.get('gross_margin'))}")
        lines.append(f"- Operating Margin: {self._fmt_pct(self.normalized.get('operating_margin'))}")
        
        roe = self.yahoo_data.get('financialData', {}).get('returnOnEquity')
        if roe:
            lines.append(f"- ROE: {self._fmt_pct(roe)}")
        lines.append("")
        
        # Growth
        lines.append("## Growth")
        lines.append(f"- Revenue Growth (YoY): {self._fmt_pct(self.normalized.get('revenue_growth_yoy'))}")
        lines.append("")
        
        # Financial health
        lines.append("## Financial Health")
        lines.append(f"- Debt/Equity: {self._fmt_num(self.normalized.get('debt_to_equity'))}")
        
        current_ratio = self.yahoo_data.get('financialData', {}).get('currentRatio')
        if current_ratio:
            lines.append(f"- Current Ratio: {self._fmt_num(current_ratio)}")
        
        fcf = self.normalized.get('free_cash_flow')
        if fcf:
            lines.append(f"- Free Cash Flow: ${self._fmt_num(fcf)}M")
        lines.append("")
        
        # Earnings history
        if self.earnings:
            lines.append("## Recent Earnings (Last 4 Quarters)")
            for e in self.earnings[:4]:
                period = e.get('period', 'N/A')
                actual = e.get('actual')
                estimate = e.get('estimate')
                if actual is not None and estimate is not None:
                    surprise = ((actual - estimate) / abs(estimate)) * 100 if estimate else 0
                    beat = "Beat" if actual > estimate else "Miss" if actual < estimate else "Met"
                    lines.append(f"- {period}: ${actual:.2f} vs ${estimate:.2f} est ({beat}, {surprise:+.1f}%)")
            lines.append("")
        
        # Analyst data
        if self.analyst_recommendations:
            latest = self.analyst_recommendations[0]
            lines.append("## Analyst Recommendations")
            lines.append(f"- Consensus: {latest.get('consensus', 'N/A')}")
            lines.append(f"- Strong Buy: {latest.get('strongBuy', 0)}, Buy: {latest.get('buy', 0)}, Hold: {latest.get('hold', 0)}, Sell: {latest.get('sell', 0)}, Strong Sell: {latest.get('strongSell', 0)}")
            lines.append("")
        
        if self.price_target:
            lines.append("## Price Targets")
            lines.append(f"- Mean: ${self._fmt_num(self.price_target.get('targetMean'))}")
            lines.append(f"- High: ${self._fmt_num(self.price_target.get('targetHigh'))}")
            lines.append(f"- Low: ${self._fmt_num(self.price_target.get('targetLow'))}")
            
            # Calculate upside
            c = self.quote.get('c') or self.quote.get('currentPrice')
            mean = self.price_target.get('targetMean')
            if c and mean:
                upside = ((mean - c) / c) * 100
                lines.append(f"- Upside to Mean: {upside:+.1f}%")
            lines.append("")
        
        # Recent news
        if self.news:
            lines.append("## Recent News")
            for n in self.news[:max_news]:
                title = n.get('title', '')[:100]
                date = (n.get('published_at') or '')[:10]
                lines.append(f"- [{date}] {title}")
            lines.append("")
        
        # Peers
        if self.peers:
            lines.append(f"## Peer Companies: {', '.join(self.peers[:8])}")
            lines.append("")
        
        # Technical context from Yahoo
        summary_detail = self.yahoo_data.get('summaryDetail', {})
        high_52w = summary_detail.get('fiftyTwoWeekHigh')
        low_52w = summary_detail.get('fiftyTwoWeekLow')
        current = self.quote.get('c') or self.yahoo_data.get('price', {}).get('regularMarketPrice')
        
        if high_52w and low_52w and current:
            pct_from_high = ((high_52w - current) / high_52w) * 100
            pct_from_low = ((current - low_52w) / low_52w) * 100
            range_position = ((current - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50
            
            lines.append("## Technical Context")
            lines.append(f"- 52-Week Range: ${low_52w:.2f} - ${high_52w:.2f}")
            lines.append(f"- Current is {pct_from_high:.1f}% below 52-week high")
            lines.append(f"- Position in 52-week range: {range_position:.0f}% (0%=low, 100%=high)")
            
            # Moving averages if available
            ma_50 = summary_detail.get('fiftyDayAverage')
            ma_200 = summary_detail.get('twoHundredDayAverage')
            if ma_50:
                above_50 = "above" if current > ma_50 else "below"
                lines.append(f"- Trading {above_50} 50-day MA (${ma_50:.2f})")
            if ma_200:
                above_200 = "above" if current > ma_200 else "below"
                lines.append(f"- Trading {above_200} 200-day MA (${ma_200:.2f})")
            lines.append("")
        
        # Data gaps
        if self.gaps:
            lines.append("## Data Gaps (Missing Information)")
            for gap in self.gaps:
                lines.append(f"- {gap}")
        
        return "\n".join(lines)

    def _fmt_num(self, val: Any) -> str:
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.2f}"
        except:
            return "N/A"

    def _fmt_pct(self, val: Any) -> str:
        if val is None:
            return "N/A"
        try:
            return f"{float(val) * 100:.1f}%" if abs(float(val)) < 1 else f"{float(val):.1f}%"
        except:
            return "N/A"


async def aggregate_stock_data(
    symbol: str,
    *,
    include_news: bool = True,
    include_peers: bool = True,
    news_days_back: int = 7,
    news_limit: int = 6,
) -> StockDataBundle:
    """
    Aggregates all available data for a symbol.
    
    This is the main entry point for gathering data before AI analysis.
    
    Usage:
        bundle = await aggregate_stock_data("AAPL")
        context = bundle.to_ai_context()
        # Pass context to your LLM
    """
    # Import here to avoid circular deps - adjust paths as needed
    from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
    from services.finnhub.finnhub_analyst import fetch_analyst_data_cached
    from services.finnhub.finnhub_news_service import get_company_news_cached
    from services.finnhub.finnhub_service import FinnhubService
    from services.yahoo_service import get_full_stock_data
    
    sym = (symbol or "").strip().upper()
    if not sym:
        return StockDataBundle(symbol="", gaps=["Missing symbol"])

    bundle = StockDataBundle(symbol=sym)
    
    # Build task list
    tasks = {
        "fundamentals": fetch_fundamentals_cached(sym),
        "analyst": fetch_analyst_data_cached(sym),
    }
    
    if include_news:
        tasks["news"] = get_company_news_cached(sym, days_back=news_days_back, limit=news_limit)
    
    # Run async tasks
    results = await asyncio.gather(
        *tasks.values(),
        return_exceptions=True
    )
    
    result_map = dict(zip(tasks.keys(), results))
    
    # Process fundamentals
    fund_result = result_map.get("fundamentals")
    if isinstance(fund_result, Exception):
        bundle.gaps.append(f"Fundamentals fetch failed: {fund_result}")
    elif hasattr(fund_result, 'data'):
        data = fund_result.data
        bundle.profile = data.get("profile", {})
        bundle.quote = data.get("quote", {})
        bundle.metrics = data.get("metrics", {})
        bundle.normalized = data.get("normalized", {})
        bundle.earnings = data.get("earnings", [])
        bundle.gaps.extend(fund_result.gaps)
    
    # Process analyst data
    analyst_result = result_map.get("analyst")
    if isinstance(analyst_result, Exception):
        bundle.gaps.append(f"Analyst data fetch failed: {analyst_result}")
    elif isinstance(analyst_result, dict):
        bundle.analyst_recommendations = analyst_result.get("recommendationHistory", [])
        bundle.price_target = analyst_result.get("priceTarget")
        bundle.gaps.extend(analyst_result.get("gaps", []))
    
    # Process news
    if include_news:
        news_result = result_map.get("news")
        if isinstance(news_result, Exception):
            bundle.gaps.append(f"News fetch failed: {news_result}")
        elif isinstance(news_result, dict):
            bundle.news = news_result.get("items", [])
        elif isinstance(news_result, list):
            bundle.news = news_result
    
    # Fetch peers - use overrides for major stocks
    if include_peers:
        if sym in PEER_OVERRIDES:
            bundle.peers = PEER_OVERRIDES[sym]
        else:
            try:
                svc = FinnhubService()
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    finnhub_peers = await svc.fetch_peers(sym, client=client)
                    bundle.peers = get_peers_for_symbol(sym, finnhub_peers)
            except Exception as e:
                bundle.gaps.append(f"Peers fetch failed: {e}")
    
    # Fetch Yahoo data (sync, run in thread)
    try:
        yahoo_result = await asyncio.to_thread(get_full_stock_data, sym)
        if isinstance(yahoo_result, dict) and yahoo_result.get("status") != "error":
            bundle.yahoo_data = yahoo_result
            
            # Extract price target from Yahoo (Finnhub price-target is premium)
            financial_data = yahoo_result.get("financialData", {})
            if financial_data.get("targetMeanPrice"):
                bundle.price_target = {
                    "targetHigh": financial_data.get("targetHighPrice"),
                    "targetLow": financial_data.get("targetLowPrice"),
                    "targetMean": financial_data.get("targetMeanPrice"),
                    "targetMedian": financial_data.get("targetMedianPrice"),
                    "numberOfAnalysts": financial_data.get("numberOfAnalystOpinions"),
                }
                # Remove the "No price target" gap if we got it from Yahoo
                bundle.gaps = [g for g in bundle.gaps if "price target" not in g.lower()]
        else:
            bundle.gaps.append("Yahoo data unavailable")
    except Exception as e:
        bundle.gaps.append(f"Yahoo fetch failed: {e}")
    
    return bundle


# Convenience function for quick context generation
async def get_analysis_context(symbol: str) -> str:
    """
    One-liner to get formatted context string for LLM.
    
    Usage:
        context = await get_analysis_context("AAPL")
        # Pass to your AI prompt
    """
    bundle = await aggregate_stock_data(symbol)
    return bundle.to_ai_context()


# ============================================================================
# MEGA-CAP PEER OVERRIDES
# ============================================================================
# Finnhub peers are industry-based (AAPL -> WDC, DELL) which isn't useful
# for investment comparison. Override for major stocks.

PEER_OVERRIDES = {
    # Mega-cap tech
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL", "NFLX"],
    "GOOG": ["META", "MSFT", "AMZN", "AAPL", "NFLX"],
    "AMZN": ["MSFT", "GOOGL", "AAPL", "META", "WMT"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT", "NFLX"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "TSM"],
    "TSLA": ["F", "GM", "RIVN", "NIO", "BYD"],
    
    # Financials
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "V": ["MA", "PYPL", "AXP", "SQ", "FIS"],
    
    # Healthcare
    "JNJ": ["PFE", "MRK", "ABBV", "UNH", "LLY"],
    "UNH": ["CVS", "ANTM", "CI", "HUM", "JNJ"],
}


def get_peers_for_symbol(symbol: str, finnhub_peers: list[str]) -> list[str]:
    """
    Get appropriate peers - use override if available, else Finnhub.
    """
    sym = symbol.upper()
    if sym in PEER_OVERRIDES:
        return PEER_OVERRIDES[sym]
    # Filter out weird Finnhub results (same symbol, obviously wrong matches)
    return [p for p in finnhub_peers if p != sym][:8]