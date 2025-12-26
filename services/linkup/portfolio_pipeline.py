"""
1. Builds portfolio metrics from user holdings
2. Calls all 4 agents:
   - news_sentiment_agent
   - performance_agent
   - scenarios_rebalance_agent
   - summary_agent  (summary)
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List
from .agents.news_sentiment_agent import call_link_up_for_news
from .agents.performance_predictions_agent import call_link_up_for_performance
from .agents.scenarios_rebalance_agent import call_link_up_for_rebalance
from .agents.summary_agent import call_link_up_for_summary
from services.linkup.metrics.build_metrics_yahoo import build_metrics, holdings_to_positions

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
        elif asset_class in ("crypto", "cryptocurrency"):
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

async def run_portfolio_pipeline(
    *,
    holdings_items: List[Any],
    base_currency: str = "USD",
    benchmark_ticker: str = "SPY",
    days_of_news: int = 7,
) -> Dict[str, Any]:
    positions = holdings_to_positions(holdings_items, base_currency=base_currency)

    metrics = await build_metrics(positions=positions, base_currency=base_currency)

    symbols = list(metrics["per_symbol"].keys())
    classification = simple_classification_from_metrics(metrics)

    # Run the 3 independent agents concurrently (they are sync -> to_thread)
    news_task = asyncio.to_thread(call_link_up_for_news, base_currency=base_currency, symbols=symbols, days_of_news=days_of_news)
    perf_task = asyncio.to_thread(call_link_up_for_performance, base_currency=base_currency, symbols=symbols, metrics=metrics, days_of_news=days_of_news)
    scen_task = asyncio.to_thread(call_link_up_for_rebalance, base_currency=base_currency, symbols=symbols, metrics=metrics, classification=classification, days_of_news=days_of_news)

    news_sentiment_json, performance_json, scenarios_json = await asyncio.gather(
        news_task, perf_task, scen_task
    )

    # Summary after the others complete
    summary_json = await asyncio.to_thread(
        call_link_up_for_summary,
        news_sentiment_json=news_sentiment_json,
        performance_predictions_json=performance_json,
        scenarios_rebalance_json=scenarios_json,
    )

    return {
        "metrics": metrics,
        "classification": classification,
        "news_sentiment": news_sentiment_json, 
        "performance": performance_json,
        "scenarios_rebalance": scenarios_json,
        "summary": summary_json,
    }