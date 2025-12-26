
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