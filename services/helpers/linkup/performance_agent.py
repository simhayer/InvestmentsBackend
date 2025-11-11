from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .linkup_config import client

PERFORMANCE_PREDICTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "performance_analysis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "leaders": {"type": "array", "items": {"type": "string"}},   # symbols
                "laggards": {"type": "array", "items": {"type": "string"}},  # symbols
                "notable_shifts": {"type": "array", "items": {"type": "string"}},
            },
        },
        "predictions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "forecast_window": {"type": "string"},  # e.g. "30D" or "90D"
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "symbol": {"type": "string"},
                            "expected_direction": {
                                "type": "string",
                                "enum": ["up", "down", "neutral"],
                            },
                            "expected_change_pct": {"type": "number"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"},
                        },
                        "required": ["symbol", "expected_direction"],
                    },
                },
            },
        },
        "explainability": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "section_confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "performance_analysis": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "predictions": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

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

def call_link_up_for_performance(base_currency, symbols, metrics):
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=7)
    try:
        response = client.search(
            query=json.dumps(build_performance_predictions_query(base_currency=base_currency, symbols=symbols, metrics=metrics)),
            depth="standard",
            output_type="structured",
            structured_output_schema=json.dumps(PERFORMANCE_PREDICTIONS_SCHEMA),
            include_images=False,
            include_sources=False,
            from_date=from_date.date(),
            to_date=to_date.date(),
        )

        return response
    except Exception as e:
        # logger.error(f"Error occurred while fetching portfolio AI layers: {e}")
        return {"error": str(e)}