from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .linkup_config import client

SCENARIOS_REBALANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scenarios": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bull": {"type": "string"},
                "base": {"type": "string"},
                "bear": {"type": "string"},
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bull": {"type": "number"},
                        "base": {"type": "number"},
                        "bear": {"type": "number"},
                    },
                },
            },
        },
        "rebalance_paths": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "aggressive_growth": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        # IMPORTANT: these "actions" are descriptive ideas, not trade instructions
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "balanced_growth": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "capital_preservation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "market_outlook": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "short_term": {"type": "string"},
                "medium_term": {"type": "string"},
                "key_opportunities": {"type": "array", "items": {"type": "string"}},
                "key_risks": {"type": "array", "items": {"type": "string"}},
            },
        },
        "actions": {
            "type": "array",
            "description": "Non-prescriptive, maintenance-style next steps.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                    "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
                    "effort": {"type": "string", "enum": ["low", "medium", "high"]},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "category": {
                        "type": "string",
                        "description": "research, diversification_review, risk_review, tax_check, liquidity_check",
                    },
                },
                "required": ["title", "rationale"],
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
                        "scenarios": {"type": "number", "minimum": 0, "maximum": 1},
                        "rebalance_paths": {"type": "number", "minimum": 0, "maximum": 1},
                        "market_outlook": {"type": "number", "minimum": 0, "maximum": 1},
                        "actions": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

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

def call_link_up_for_rebalance(base_currency, symbols, metrics, classification):
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=7)
    try:
        response = client.search(
            query=json.dumps(build_scenarios_rebalance_query(base_currency=base_currency, symbols=symbols, metrics=metrics,
                                                  classification=classification)),
            depth="standard",
            output_type="structured",
            structured_output_schema=json.dumps(SCENARIOS_REBALANCE_SCHEMA),
            include_images=False,
            include_sources=False,
            from_date=from_date.date(),
            to_date=to_date.date(),
        )

        return response
    except Exception as e:
        # logger.error(f"Error occurred while fetching portfolio AI layers: {e}")
        return {"error": str(e)}
