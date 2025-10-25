from datetime import datetime, timedelta, timezone
import json
from .linkup_config import client

to_date = datetime.now(timezone.utc)
from_date = to_date - timedelta(days=7) 

schema = {
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

def get_linkup_market_summary() -> dict:
    """Fetches and returns the LinkUp market summary response."""
    response = client.search(
    query=(
        "You are a financial analyst. Summarize today's key US market developments "
        "in a 'cause → impact' format. For each event, explain what happened, why it happened, "
        "and how it affected financial markets. Focus on recent data and news from the past week "
        "(S&P 500, NASDAQ, Dow Jones, yields, CPI, Fed policy, energy, etc.). "
        "Use reliable sources like Bloomberg, Reuters, and WSJ."
        ),
        depth="standard",
        output_type="structured",
        structured_output_schema=json.dumps(schema),
        include_images=False,
        include_sources=False, 
        from_date=from_date.date(),
        to_date=to_date.date()
    )
    return response
