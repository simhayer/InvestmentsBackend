# portfolio_analysis_aggregator.py
"""
Aggregates portfolio data for AI analysis.
Combines holdings, prices, fundamentals, and news for comprehensive portfolio review.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict


@dataclass
class PortfolioDataBundle:
    """All data needed for AI portfolio analysis."""
    user_id: str
    currency: str
    
    # Holdings data
    holdings: List[Dict[str, Any]] = field(default_factory=list)
    top_holdings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregated metrics
    total_value: float = 0.0
    total_cost: float = 0.0
    total_pl: float = 0.0
    total_pl_pct: float = 0.0
    day_pl: float = 0.0
    day_pl_pct: float = 0.0
    
    # Allocation breakdowns
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    type_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Per-symbol fundamentals (for top holdings)
    symbol_fundamentals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recent news for holdings
    portfolio_news: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk metrics
    concentration_risk: float = 0.0  # Top 3 holdings as % of portfolio
    position_count: int = 0
    
    # Data quality
    gaps: List[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "userId": self.user_id,
            "currency": self.currency,
            "holdings": self.holdings,
            "topHoldings": self.top_holdings,
            "totalValue": self.total_value,
            "totalCost": self.total_cost,
            "totalPL": self.total_pl,
            "totalPLPct": self.total_pl_pct,
            "dayPL": self.day_pl,
            "dayPLPct": self.day_pl_pct,
            "sectorAllocation": self.sector_allocation,
            "typeAllocation": self.type_allocation,
            "symbolFundamentals": self.symbol_fundamentals,
            "portfolioNews": self.portfolio_news,
            "concentrationRisk": self.concentration_risk,
            "positionCount": self.position_count,
            "gaps": self.gaps,
            "fetchedAt": self.fetched_at,
        }

    def to_ai_context(self, max_news: int = 8) -> str:
        """Format portfolio data for LLM analysis."""
        lines = ["# Portfolio Analysis Data", ""]
        
        # Portfolio Overview
        lines.append("## Portfolio Overview")
        lines.append(f"- Total Value: ${self.total_value:,.2f} {self.currency}")
        lines.append(f"- Total Cost Basis: ${self.total_cost:,.2f}")
        lines.append(f"- Total P/L: ${self.total_pl:+,.2f} ({self.total_pl_pct:+.2f}%)")
        lines.append(f"- Today's Change: ${self.day_pl:+,.2f} ({self.day_pl_pct:+.2f}%)")
        lines.append(f"- Number of Positions: {self.position_count}")
        lines.append("")
        
        # Risk Metrics
        lines.append("## Risk Metrics")
        lines.append(f"- Concentration Risk (Top 3): {self.concentration_risk:.1f}%")
        
        diversification = "Well Diversified" if self.concentration_risk < 40 else \
                         "Moderately Concentrated" if self.concentration_risk < 60 else \
                         "Highly Concentrated"
        lines.append(f"- Diversification: {diversification}")
        lines.append("")
        
        # Asset Type Allocation
        if self.type_allocation:
            lines.append("## Asset Type Allocation")
            for asset_type, pct in sorted(self.type_allocation.items(), key=lambda x: -x[1]):
                lines.append(f"- {asset_type}: {pct:.1f}%")
            lines.append("")
        
        # Sector Allocation
        if self.sector_allocation:
            lines.append("## Sector Allocation")
            for sector, pct in sorted(self.sector_allocation.items(), key=lambda x: -x[1]):
                lines.append(f"- {sector}: {pct:.1f}%")
            lines.append("")
        
        # Top Holdings (with industry from fundamentals when available)
        if self.top_holdings:
            lines.append("## Top Holdings")
            for h in self.top_holdings[:10]:
                symbol = h.get("symbol", "?")
                name = h.get("name", "")[:30]
                weight = h.get("weight", 0)
                pl_pct = h.get("unrealized_pl_pct", 0) or 0
                value = h.get("current_value", 0) or 0
                
                # Look up industry from fundamentals profile
                industry = ""
                fund = self.symbol_fundamentals.get(symbol, {})
                if fund:
                    industry = (fund.get("profile", {}).get("finnhubIndustry") or "").strip()
                
                industry_tag = f" [{industry}]" if industry else ""
                pl_sign = "+" if pl_pct >= 0 else ""
                lines.append(f"- {symbol} ({name}){industry_tag}: {weight:.1f}% of portfolio, ${value:,.0f}, P/L: {pl_sign}{pl_pct:.1f}%")
            lines.append("")
        
        # Per-Symbol Fundamentals
        if self.symbol_fundamentals:
            lines.append("## Key Fundamentals (Top Holdings)")
            for symbol, fund in self.symbol_fundamentals.items():
                profile = fund.get("profile", {})
                normalized = fund.get("normalized", {})
                
                industry = (profile.get("finnhubIndustry") or "").strip()
                pe = normalized.get("pe_ttm")
                market_cap = normalized.get("market_cap")
                margin = normalized.get("operating_margin")
                growth = normalized.get("revenue_growth_yoy")
                debt_eq = normalized.get("debt_to_equity")
                
                parts = [f"{symbol}:"]
                if industry: parts.append(f"Industry: {industry}")
                if market_cap: parts.append(f"MCap ${market_cap/1e3:.1f}B" if market_cap >= 1e3 else f"MCap ${market_cap:.0f}M")
                if pe: parts.append(f"P/E {pe:.1f}")
                if margin: parts.append(f"Margin {margin*100:.1f}%")
                if growth: parts.append(f"Growth {growth:.1f}%")
                if debt_eq is not None: parts.append(f"D/E {debt_eq:.2f}")
                
                if len(parts) > 1:
                    lines.append(f"- {' | '.join(parts)}")
            lines.append("")
        
        # Winners and Losers
        sorted_by_pl = sorted(
            [h for h in self.holdings if h.get("unrealized_pl_pct") is not None],
            key=lambda x: x.get("unrealized_pl_pct", 0),
            reverse=True
        )
        
        if sorted_by_pl:
            winners = sorted_by_pl[:3]
            losers = sorted_by_pl[-3:]
            
            lines.append("## Best Performers")
            for h in winners:
                if (h.get("unrealized_pl_pct") or 0) > 0:
                    lines.append(f"- {h['symbol']}: +{h['unrealized_pl_pct']:.1f}%")
            
            lines.append("")
            lines.append("## Worst Performers")
            for h in reversed(losers):
                if (h.get("unrealized_pl_pct") or 0) < 0:
                    lines.append(f"- {h['symbol']}: {h['unrealized_pl_pct']:.1f}%")
            lines.append("")
        
        # Recent News
        if self.portfolio_news:
            lines.append("## Recent News (Portfolio Companies)")
            for n in self.portfolio_news[:max_news]:
                symbol = n.get("symbol", "")
                title = n.get("title", "")[:80]
                date = (n.get("published_at") or "")[:10]
                lines.append(f"- [{symbol}] {date}: {title}")
            lines.append("")
        
        # Data Gaps
        if self.gaps:
            lines.append("## Data Gaps")
            for gap in self.gaps:
                lines.append(f"- {gap}")
        
        return "\n".join(lines)


async def aggregate_portfolio_data(
    user_id: str,
    db,  # SQLAlchemy Session
    finnhub,  # FinnhubService instance
    *,
    currency: str = "USD",
    include_fundamentals: bool = True,
    include_news: bool = True,
    top_n: int = 10,
) -> PortfolioDataBundle:
    """
    Aggregates all portfolio data for AI analysis.
    
    Args:
        user_id: The user's ID
        db: Database session
        finnhub: FinnhubService instance
        currency: Target currency for values
        include_fundamentals: Whether to fetch fundamentals for top holdings
        include_news: Whether to fetch news for holdings
        top_n: Number of top holdings to analyze in detail
    """
    # Import here to avoid circular deps
    from services.holding_service import get_holdings_with_live_prices
    from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
    from services.finnhub.finnhub_news_service import get_company_news_cached
    
    bundle = PortfolioDataBundle(user_id=user_id, currency=currency)
    
    # 1. Fetch holdings with prices
    try:
        holdings_data = await get_holdings_with_live_prices(
            user_id,
            db,
            finnhub,
            currency=currency,
            include_weights=True,
        )
        
        items = holdings_data.get("items", [])
        bundle.holdings = [_holding_to_dict(h) for h in items]
        bundle.top_holdings = [_holding_to_dict(h) for h in holdings_data.get("top_items", [])]
        bundle.total_value = holdings_data.get("market_value", 0)
        bundle.position_count = len(items)
        
    except Exception as e:
        bundle.gaps.append(f"Holdings fetch failed: {e}")
        return bundle
    
    if not bundle.holdings:
        bundle.gaps.append("No holdings found")
        return bundle
    
    # 2. Calculate aggregated metrics
    _calculate_portfolio_metrics(bundle)
    
    # 3. Calculate allocations
    _calculate_allocations(bundle)
    
    # 4. Fetch fundamentals for top holdings
    if include_fundamentals and bundle.top_holdings:
        top_symbols = [h["symbol"] for h in bundle.top_holdings[:top_n] if h.get("symbol")]
        
        try:
            fund_tasks = [fetch_fundamentals_cached(sym) for sym in top_symbols]
            fund_results = await asyncio.gather(*fund_tasks, return_exceptions=True)
            
            for sym, result in zip(top_symbols, fund_results):
                if isinstance(result, Exception):
                    continue
                if hasattr(result, 'data') and result.data:
                    bundle.symbol_fundamentals[sym] = result.data
                    
        except Exception as e:
            bundle.gaps.append(f"Fundamentals fetch failed: {e}")
    
    # 5. Calculate sector allocation (requires fundamentals from step 4)
    if bundle.symbol_fundamentals:
        _calculate_sector_allocation(bundle)
    
    # 6. Fetch news for holdings
    if include_news and bundle.top_holdings:
        news_symbols = [h["symbol"] for h in bundle.top_holdings[:5] if h.get("symbol")]
        
        try:
            news_tasks = [
                get_company_news_cached(sym, days_back=7, limit=3) 
                for sym in news_symbols
            ]
            news_results = await asyncio.gather(*news_tasks, return_exceptions=True)
            
            all_news = []
            for sym, result in zip(news_symbols, news_results):
                if isinstance(result, Exception):
                    continue
                if isinstance(result, dict):
                    items = result.get("items", [])
                elif isinstance(result, list):
                    items = result
                else:
                    continue
                    
                for item in items:
                    item["symbol"] = sym
                    all_news.append(item)
            
            # Sort by date, newest first
            all_news.sort(
                key=lambda x: x.get("published_at", ""),
                reverse=True
            )
            bundle.portfolio_news = all_news[:15]
            
        except Exception as e:
            bundle.gaps.append(f"News fetch failed: {e}")
    
    return bundle


def _holding_to_dict(h) -> Dict[str, Any]:
    """Convert HoldingOut to dict."""
    if isinstance(h, dict):
        return h
    return {
        "id": getattr(h, "id", None),
        "symbol": getattr(h, "symbol", ""),
        "name": getattr(h, "name", ""),
        "type": getattr(h, "type", "equity"),
        "quantity": getattr(h, "quantity", 0),
        "current_price": getattr(h, "current_price", 0),
        "current_value": getattr(h, "current_value", 0),
        "purchase_unit_price": getattr(h, "purchase_unit_price", 0),
        "purchase_amount_total": getattr(h, "purchase_amount_total", 0),
        "unrealized_pl": getattr(h, "unrealized_pl", 0),
        "unrealized_pl_pct": getattr(h, "unrealized_pl_pct", 0),
        "day_pl": getattr(h, "day_pl", 0),
        "weight": getattr(h, "weight", 0),
        "currency": getattr(h, "currency", "USD"),
        "institution": getattr(h, "institution", ""),
        "account_name": getattr(h, "account_name", ""),
    }


def _calculate_portfolio_metrics(bundle: PortfolioDataBundle) -> None:
    """Calculate aggregate P/L metrics."""
    total_cost = 0.0
    total_pl = 0.0
    day_pl = 0.0
    
    for h in bundle.holdings:
        cost = h.get("purchase_amount_total") or 0
        pl = h.get("unrealized_pl") or 0
        day = h.get("day_pl") or 0
        
        total_cost += cost
        total_pl += pl
        day_pl += day
    
    bundle.total_cost = total_cost
    bundle.total_pl = total_pl
    bundle.total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    bundle.day_pl = day_pl
    bundle.day_pl_pct = (day_pl / bundle.total_value * 100) if bundle.total_value > 0 else 0
    
    # Concentration risk: top 3 as % of total
    weights = sorted([h.get("weight", 0) for h in bundle.holdings], reverse=True)
    bundle.concentration_risk = sum(weights[:3])


def _calculate_allocations(bundle: PortfolioDataBundle) -> None:
    """Calculate asset type allocations from holdings."""
    type_totals = defaultdict(float)
    
    for h in bundle.holdings:
        weight = h.get("weight", 0)
        
        # Asset type
        asset_type = (h.get("type") or "equity").lower()
        type_label = {
            "equity": "Stocks",
            "stock": "Stocks",
            "etf": "ETFs",
            "mutual_fund": "Mutual Funds",
            "crypto": "Crypto",
            "cryptocurrency": "Crypto",
            "bond": "Bonds",
            "cash": "Cash",
        }.get(asset_type, "Other")
        type_totals[type_label] += weight
    
    bundle.type_allocation = dict(type_totals)


def _calculate_sector_allocation(bundle: PortfolioDataBundle) -> None:
    """
    Calculate sector/industry allocation using finnhubIndustry from fundamentals.
    Must be called after fundamentals have been fetched into bundle.symbol_fundamentals.
    """
    # Build symbol -> industry map from fetched fundamentals
    symbol_industry: Dict[str, str] = {}
    for sym, fund_data in bundle.symbol_fundamentals.items():
        profile = fund_data.get("profile", {})
        industry = (profile.get("finnhubIndustry") or "").strip()
        if industry:
            symbol_industry[sym] = industry

    if not symbol_industry:
        return

    sector_totals = defaultdict(float)
    uncovered_weight = 0.0

    for h in bundle.holdings:
        symbol = (h.get("symbol") or "").upper()
        weight = h.get("weight", 0)
        asset_type = (h.get("type") or "equity").lower()

        # Skip non-equity types — crypto, cash, bonds don't have a Finnhub industry
        if asset_type in ("crypto", "cryptocurrency", "cash", "bond"):
            continue

        industry = symbol_industry.get(symbol)
        if industry:
            sector_totals[industry] += weight
        else:
            uncovered_weight += weight

    # Only include "Other" bucket if there's meaningful uncovered weight
    if uncovered_weight > 1.0:
        sector_totals["Other / Unclassified"] += uncovered_weight

    bundle.sector_allocation = dict(sector_totals)


# Convenience function
async def get_portfolio_analysis_context(
    user_id: str,
    db,
    finnhub,
    currency: str = "USD",
) -> str:
    """One-liner to get formatted context for LLM."""
    bundle = await aggregate_portfolio_data(user_id, db, finnhub, currency=currency)
    return bundle.to_ai_context()