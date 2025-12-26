
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