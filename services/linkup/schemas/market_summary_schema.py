
MARKET_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "as_of": {
            "type": "string",
            "description": "ISO-8601 timestamp (UTC)"
        },
        "market": {
            "type": "string",
            "description": "e.g., 'US'"
        },
        "sections": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {
                        "type": "string",
                        "description": "Concise title of the market event or theme"
                    },
                    "cause": {
                        "type": "string",
                        "description": "Summary of what caused or triggered the event (economic report, policy decision, earnings, etc.)"
                    },
                    "impact": {
                        "type": "string",
                        "description": "Explanation of how this event affected the financial markets (indices, sectors, FX, bonds, etc.)"
                    },
                    "affected_assets": {
                        "type": "array",
                        "description": "Key stocks, indices, or sectors most impacted",
                        "items": {"type": "string"}
                    },
                    "sources": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 5,
                        "items": {"type": "string", "format": "uri"}
                    }
                },
                "required": ["headline", "cause", "impact", "sources"]
            }
        }
    },
    "required": ["as_of", "market", "sections"]
}