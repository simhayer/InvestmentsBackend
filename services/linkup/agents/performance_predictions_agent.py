from __future__ import annotations
from services.linkup.linkup_search import linkup_structured_search
from services.linkup.schemas.performence_predictions_schema import PERFORMANCE_PREDICTIONS_SCHEMA

def build_performance_predictions_query(base_currency, symbols, metrics):
    """
    metrics: your deterministic summary per symbol, e.g.
    {
      "AAPL": {"weight_pct": 12.3, "return_1Y_pct": 18.4, "vol_1Y_pct": 22.1, ...},
      ...
    }
    """
    return {
        "role": (
            "You are a professional portfolio analyst. "
            "You ONLY interpret the deterministic metrics provided in 'metrics' and recent context from Linkup. "
            "You must output ONLY valid JSON that matches PERFORMANCE_PREDICTIONS_SCHEMA."
        ),
        "step_1_task": [
            "Identify leaders and laggards based on the provided metrics (returns, drawdowns, risk).",
            "Describe notable shifts such as concentration risk or sharp performance changes.",
            "Optionally provide a short-horizon directional outlook for a small subset of key symbols, "
            "clearly labeling it as uncertain and informational-only.",
        ],
        "step_2_context": [
            "Do NOT recompute or invent numerical metrics; rely strictly on the 'metrics' object.",
            "Use relative language: 'has outperformed the rest of the portfolio' rather than new numbers.",
            "If predictions are too speculative or unsupported, prefer fewer assets and lower confidence.",
        ],
        "step_3_references": [
            "You may call out broad macro/thematic context (e.g., 'AI-related names have benefited from ...') "
            "if supported by Linkup sources.",
            "Include citations like 【source†L#-L#】 in rationales where you lean on external information.",
        ],
        "step_4_evaluate": [
            "Ensure leaders and laggards align with the direction of returns in 'metrics'.",
            "Do NOT suggest transactions or portfolio changes; only describe patterns and risks.",
            "For predictions, avoid specific price targets, levels, or timing recommendations.",
        ],
        "step_5_iterate": [
            "If any statement implicitly sounds like 'buy/sell/add/trim/hold', rewrite it into a neutral observation.",
            "Document uncertainties, e.g. 'Short-term outlook is highly sensitive to macro data releases'.",
        ],
        "constraints": [
            "STRICTLY conform to PERFORMANCE_PREDICTIONS_SCHEMA.",
            "Do NOT tell the user to buy, sell, add, trim, or hold any asset.",
            "Do NOT invent new metrics or change metric values.",
            "Predictions are optional; leave them minimal or empty if confidence is low.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
            "metrics": metrics,
        },
    }

def call_link_up_for_performance(base_currency, symbols, metrics, days_of_news=7):
    return linkup_structured_search(
        query_obj=build_performance_predictions_query(
            base_currency=base_currency,
            symbols=symbols,
            metrics=metrics,
        ),
        schema=PERFORMANCE_PREDICTIONS_SCHEMA,
        days=days_of_news,
        include_sources=False,
    )