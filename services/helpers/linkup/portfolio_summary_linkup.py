"""
run_portfolio_pipeline.py

Orchestrator that:
1. Builds portfolio metrics from dummy Plaid positions.
2. Calls all 4 agents:
   - news_sentiment_agent
   - performance_agent
   - scenarios_rebalance_agent
   - summarry_agennt  (summary)
3. Prints the combined result.

Adjust import names / signatures to match your actual modules.
"""

from __future__ import annotations

from typing import Any, Dict, List

# -----------------------------------------------------------
# Imports: adjust to your actual function names
# -----------------------------------------------------------

# Example: each agent exposes a single `run_*` function that returns JSON.
# Change these if your file exports are different.
from .news_sentiment_agent import call_link_up_for_news
from .performance_agent import call_link_up_for_performance
from .scenarios_rebalance_agent import call_link_up_for_rebalance
from .summary_agent import call_link_up_for_summary

# Portfolio metrics – either a function or a class.
# Option A: function
from ..portfolio_metrics import build_metrics_from_plaid

# If you instead have a class, comment the above and use:
# from portfolio_metrics import PortfolioMetrics


# -----------------------------------------------------------
# 1. Dummy Plaid positions for testing
# -----------------------------------------------------------

DUMMY_PLAID_POSITIONS: List[Dict[str, Any]] = [
    {
        "symbol": "AAPL",              # US equity
        "quantity": 25,
        "cost_basis": 7500.00,         # in base_currency (e.g. CAD)
        "name": "Apple Inc.",
        "asset_class": "equity",
        "sector": "Information Technology",
        "region": "US",
    },
    {
        "symbol": "NVDA",              # US equity, growth/AI
        "quantity": 10,
        "cost_basis": 5200.00,
        "name": "NVIDIA Corporation",
        "asset_class": "equity",
        "sector": "Information Technology",
        "region": "US",
    },
    {
        "symbol": "MSFT",              # US mega-cap
        "quantity": 15,
        "cost_basis": 6000.00,
        "name": "Microsoft Corporation",
        "asset_class": "equity",
        "sector": "Information Technology",
        "region": "US",
    },
    {
        "symbol": "VFV.TO",            # TSX S&P 500 ETF (CAD)
        "quantity": 30,
        "cost_basis": 3600.00,
        "name": "Vanguard S&P 500 Index ETF",
        "asset_class": "etf",
        "sector": "Multi-sector",
        "region": "Canada",
    },
    {
        "symbol": "SHOP.TO",           # Canadian growth equity
        "quantity": 12,
        "cost_basis": 1800.00,
        "name": "Shopify Inc.",
        "asset_class": "equity",
        "sector": "Information Technology",
        "region": "Canada",
    },
    {
        "symbol": "BTC-USD",           # Crypto (priced in USD)
        "quantity": 0.15,
        "cost_basis": 5500.00,
        "name": "Bitcoin",
        "asset_class": "crypto",
        "sector": "Crypto",
        "region": "Global",
    },
    {
        "symbol": "CASH",              # Cash “position”
        "quantity": 1,
        "cost_basis": 2000.00,
        "name": "Cash Balance",
        "asset_class": "cash",
        "sector": "Cash",
        "region": "Canada",
    },
]


# -----------------------------------------------------------
# 2. Optional: simple classification helper
# -----------------------------------------------------------

def simple_classification_from_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Very rough demo classification:

      - ETFs and diversified names -> 'core'
      - Crypto and very volatile stuff -> 'speculative'
      - Cash -> 'hedge'

    You can replace this with your own logic or a dedicated classification agent.
    """
    classifications: Dict[str, str] = {}
    per_symbol = metrics["per_symbol"]

    for sym, s in per_symbol.items():
        asset_class = s.get("asset_class", "")
        mdd = s.get("max_drawdown_1Y_pct")

        if asset_class == "cash" or sym == "CASH":
            classifications[sym] = "hedge"
        elif asset_class == "etf":
            classifications[sym] = "core"
        elif asset_class == "crypto":
            classifications[sym] = "speculative"
        else:
            # crude: high max drawdown -> speculative
            if mdd is not None and mdd < -40:
                classifications[sym] = "speculative"
            else:
                classifications[sym] = "core"

    return classifications


# -----------------------------------------------------------
# 3. Orchestrator
# -----------------------------------------------------------

def run_portfolio_pipeline() -> Dict[str, Any]:
    base_currency = "CAD"
    benchmark_ticker = "SPY"

    # --- Metrics from Plaid + Yahoo ---
    # If you have a class instead:
    # metrics = PortfolioMetrics(DUMMY_PLAID_POSITIONS, base_currency, benchmark_ticker).to_dict()
    metrics = build_metrics_from_plaid(
        plaid_positions=DUMMY_PLAID_POSITIONS,
        base_currency=base_currency,
        benchmark_ticker=benchmark_ticker,
    )

    symbols = list(metrics["per_symbol"].keys())
    classification = simple_classification_from_metrics(metrics)

    # --- Agent 1: news & sentiment ---
    # Expected signature in news_sentiment_agent.py:
    # def run_news_sentiment_agent(base_currency: str, symbols: List[str]) -> Dict[str, Any]:
    news_sentiment_json = call_link_up_for_news(
        base_currency=base_currency,
        symbols=symbols,
    )
    # --- Agent 2: performance & predictions ---
    # Expected signature in performance_agent.py:
    # def run_performance_agent(base_currency: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    performance_json = call_link_up_for_performance(
        base_currency=base_currency,
        symbols=symbols,
        metrics=metrics,
    )

    # --- Agent 3: scenarios, rebalance paths, market outlook, actions ---
    # Expected signature in scenarios_rebalance_agent.py:
    # def run_scenarios_rebalance_agent(base_currency: str,
    #                                   symbols: List[str],
    #                                   metrics: Dict[str, Any],
    #                                   classification: Dict[str, str]) -> Dict[str, Any]:
    scenarios_json = call_link_up_for_rebalance(
        base_currency=base_currency,
        symbols=symbols,
        metrics=metrics,
        classification=classification,
    )

    # --- Agent 4: summary & disclaimer ---
    # Expected signature in summarry_agennt.py:
    # def run_summary_agent(news_json: Dict[str, Any],
    #                       performance_json: Dict[str, Any],
    #                       scenarios_json: Dict[str, Any]) -> Dict[str, Any]:
    summary_json = call_link_up_for_summary(
        news_sentiment_json=news_sentiment_json,
        performance_predictions_json=performance_json,
        scenarios_rebalance_json=scenarios_json,
    )

    # --- Combine everything into one object for your app ---
    final_payload: Dict[str, Any] = {
        "metrics": metrics,
        "classification": classification,
        "news_sentiment": news_sentiment_json,
        "performance": performance_json,
        "scenarios_rebalance": scenarios_json,
        "summary": summary_json,
    }

    return final_payload
