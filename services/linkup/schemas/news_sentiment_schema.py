
NEWS_SENTIMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "latest_developments": {
            "type": "array",
            "minItems": 0,
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string"},  # ISO-8601 or YYYY-MM-DD
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                    "cause": {"type": "string"},
                    "impact": {"type": "string"},
                    "assets_affected": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["headline", "date"],
            },
        },
        "catalysts": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "date": {"type": "string"},
                    "type": {"type": "string"},  # earnings, macro, product, etc.
                    "description": {"type": "string"},
                    "expected_direction": {
                        "type": "string",
                        "enum": ["up", "down", "unclear", "neutral"],
                    },
                    "magnitude_basis": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "assets_affected": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "description"],
            },
        },
        "risks_list": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "risk": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "monitor": {"type": "string"},
                    "assets_affected": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["risk", "why_it_matters"],
            },
        },
        "sentiment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_sentiment": {
                    "type": "string",
                    "enum": ["bullish", "neutral", "bearish"],
                },
                "sources_considered": {"type": "array", "items": {"type": "string"}},
                "drivers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "theme": {"type": "string"},
                            "tone": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"],
                            },
                            "impact": {"type": "string"},
                        },
                        "required": ["theme"],
                    },
                },
                "summary": {"type": "string"},
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
                        "news": {"type": "number", "minimum": 0, "maximum": 1},
                        "catalysts": {"type": "number", "minimum": 0, "maximum": 1},
                        "sentiment": {"type": "number", "minimum": 0, "maximum": 1},
                        "risks_list": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
