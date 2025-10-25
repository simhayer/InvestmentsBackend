from datetime import datetime, timedelta, timezone
import json
from .linkup_config import client

to_date = datetime.now(timezone.utc)
from_date = to_date - timedelta(days=7) 

schema = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "latest_developments": {
      "type": "array",
      "minItems": 3,
      "maxItems": 10,
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "headline": { "type": "string" },
          "date": { "type": "string", "description": "ISO-8601 UTC" },
          "source": { "type": "string" },
          "url": { "type": "string" },
          "cause": { "type": "string", "description": "What happened & why" },
          "impact": { "type": "string", "description": "Market reaction / likely effect" },
          "assets_affected": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    },
    "catalysts": {
      "type": "array",
      "maxItems": 10,
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "date": { "type": "string" },
          "type": { "type": "string", "description": "e.g., earnings, product, macro" },
          "description": { "type": "string" },
          "expected_direction": {
            "type": "string",
            "enum": ["up", "down", "unclear"]
          },
          "magnitude_basis": {
            "type": "string",
            "description": "What this depends on"
          },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "risks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "risk": { "type": "string" },
          "why_it_matters": { "type": "string" },
          "monitor": { "type": "string" }
        }
      }
    },
    "valuation": {
      "type": "object",
      "properties": {
        "multiples": {
          "type": "object",
          "properties": {
            "pe_ttm": { "type": "number" },
            "fwd_pe": { "type": "number" },
            "ps_ttm": { "type": "number" },
            "ev_ebitda": { "type": "number" }
          }
        },
        "peer_set": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "technicals": {
      "type": "object",
      "properties": {
        "trend": { "type": "string", "description": "Uptrend/Downtrend/Sideways" },
        "levels": {
          "type": "object",
          "properties": {
            "support": { "type": "number" },
            "resistance": { "type": "number" }
          }
        },
        "momentum": {
          "type": "object",
          "properties": {
            "rsi": { "type": "number" },
            "comment": { "type": "string" }
          }
        }
      }
    },
    "key_dates": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": { "type": "string" },
          "event": { "type": "string" }
        }
      }
    },
    "scenarios": {
      "type": "object",
      "properties": {
        "bull": { "type": "string" },
        "base": { "type": "string" },
        "bear": { "type": "string" }
      }
    },
    "summary": { "type": "string" },
    "disclaimer": { "type": "string" }
  }
}

def get_linkup_symbol_analysis(symbol: str) -> dict:
    """Fetches and returns the LinkUp symbol summary response."""
    response = client.search(
    query=f"Analyse the {symbol} stock for me, act as a financial advisor, tell me latest happenings, updates, events, factors that could affect the stock, and also predictions on some basis",
        depth="standard",
        output_type="structured",
        structured_output_schema=json.dumps(schema),
        include_images=False,
        include_sources=False, 
        from_date=from_date.date(),
        to_date=to_date.date()
    )
    return response
