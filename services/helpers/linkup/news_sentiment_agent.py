from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .linkup_config import client

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

def build_news_sentiment_query(base_currency, symbols):
    return {
        "role": (
            "You are a professional portfolio analyst using Linkup as your research engine. "
            "You must output ONLY valid JSON that matches the NEWS_SENTIMENT_SCHEMA. "
            "You are not allowed to give trading instructions or recommendations."
        ),
        "step_1_task": [
            "Scan recent, credible sources for news and events related to the provided symbols and major macro drivers.",
            "Summarize only well-supported headlines and catalysts; avoid fabricating dates, numbers, or company actions.",
            "Identify portfolio-relevant risks and monitoring points (company-specific and macro).",
            "Provide an aggregated sentiment view (bullish/neutral/bearish) with narrative drivers.",
        ],
        "step_2_context": [
            "Assume North American markets (NASDAQ, NYSE, TSX) as the primary context.",
            "Use Linkup search results and Yahoo Finance-style fundamentals where available.",
            "If there is little or no recent news for a symbol, prefer omitting it over guessing.",
        ],
        "step_3_references": [
            "Use only credible sources returned by Linkup.",
            "When summarizing a source, include inline citations in the form 【source†L#-L#】 in text fields.",
            "Do NOT invent headlines, event dates, or percentage moves. If uncertain, state 'unclear' or omit the item.",
        ],
        "step_4_evaluate": [
            "Check that each headline and catalyst could realistically be supported by at least one source.",
            "If support is weak or ambiguous, lower the relevant section_confidence and mention it in explainability.limitations.",
            "Avoid any language that tells the user to buy, sell, add, trim, or hold positions.",
        ],
        "step_5_iterate": [
            "After drafting, remove any speculative statements that are not clearly supported by sources.",
            "Prefer fewer, well-supported developments over many low-confidence ones.",
        ],
        "constraints": [
            "Output must be STRICTLY valid JSON per NEWS_SENTIMENT_SCHEMA.",
            "Populate assets_affected ONLY with symbols from the provided list.",
            "All content is informational and descriptive, not investment advice.",
            "Do NOT use imperative verbs like 'buy', 'sell', 'add', 'trim', or 'hold'.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
        },
    }


def call_link_up_for_news(base_currency, symbols):
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=7)
    try:
        response = client.search(
            query=json.dumps(build_news_sentiment_query(base_currency=base_currency, symbols=symbols)),
            depth="standard",
            output_type="structured",
            structured_output_schema=json.dumps(NEWS_SENTIMENT_SCHEMA),
            include_images=False,
            include_sources=False,
            from_date=from_date.date(),
            to_date=to_date.date(),
        )

        return response
    except Exception as e:
        # logger.error(f"Error occurred while fetching portfolio AI layers: {e}")
        return {"error": str(e)}