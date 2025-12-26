from __future__ import annotations
from services.linkup.linkup_search import linkup_structured_search
from services.linkup.schemas.scenarios_rebalance_schema import SCENARIOS_REBALANCE_SCHEMA

def build_scenarios_rebalance_query(base_currency, symbols, metrics, classification):
    """
    classification: mapping symbol -> 'core' | 'speculative' | 'hedge'
    metrics: deterministic concentration, volatility, etc. (read-only)
    """
    return {
        "role": (
            "You are a professional portfolio strategist. "
            "You speak in frameworks and scenario language, NOT in concrete trade instructions. "
            "Output ONLY valid JSON matching SCENARIOS_REBALANCE_SCHEMA."
        ),
        "step_1_task": [
            "Define high-level bull/base/bear scenarios that are relevant to the overall portfolio, "
            "not to a single ticker.",
            "For each risk profile (aggressive_growth, balanced_growth, capital_preservation), "
            "describe how a typical investor might THINK about adjusting concentrations, "
            "without telling the user to execute trades.",
            "List non-prescriptive actions the user could take, such as reviews, research tasks, "
            "or checks (e.g., 'Review your exposure to single-name tech positions > X% of portfolio').",
        ],
        "step_2_context": [
            "Use 'core/speculative/hedge' classification and concentration metrics as inputs.",
            "Avoid explicit words like buy, sell, add, trim, increase position, or reduce position.",
            "Instead use language like 'high reliance on', 'significant exposure to', "
            "and 'may want to review whether this aligns with your risk tolerance'.",
        ],
        "step_3_references": [
            "Ground market_outlook in broad, well-known macro and sector trends when available via Linkup.",
            "Include citations in free-text fields when referencing external macro or sector commentary.",
        ],
        "step_4_evaluate": [
            "Ensure that none of the summaries or actions can be reasonably interpreted as direct trading advice.",
            "Keep probabilities in scenarios rough and clearly subjective (e.g., 0.2 / 0.5 / 0.3).",
            "If you are unsure about a scenario, lower section_confidence and describe the uncertainty.",
        ],
        "step_5_iterate": [
            "Rewrite any remaining prescriptive sentences into neutral, informational descriptions or questions.",
            "Prefer phrasing like 'Consider discussing with a professional advisor if...' rather than instructions.",
        ],
        "constraints": [
            "STRICTLY conform to SCENARIOS_REBALANCE_SCHEMA.",
            "Do NOT use words: 'buy', 'sell', 'add', 'trim', 'hold', 'rebalance by X%', "
            "or any explicit order to change positions.",
            "All content is general and informational, not tailored financial advice.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
            "metrics": metrics
        },
    }

def call_link_up_for_rebalance(base_currency, symbols, metrics, classification, days_of_news=7):
    return linkup_structured_search(
        query_obj=build_scenarios_rebalance_query(
            base_currency=base_currency,
            symbols=symbols,
            metrics=metrics,
            classification=classification,
        ),
        schema=SCENARIOS_REBALANCE_SCHEMA,
        days=days_of_news,
        include_sources=False,
    )
