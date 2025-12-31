from typing import Any, Dict

SINGLE_STOCK_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "symbol": {"type": "string"},
        "company_profile": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "sector": {"type": ["string", "null"]},
                "industry": {"type": ["string", "null"]},
                "country": {"type": ["string", "null"]},
                "business_model": {"type": ["string", "null"]},
                "key_products_services": {"type": "array", "items": {"type": "string"}},
                "revenue_drivers": {"type": "array", "items": {"type": "string"}},
                "moat_and_competitive_advantages": {"type": ["string", "null"]},
                "cyclicality": {"type": ["string", "null"]},
                "capital_intensity": {"type": ["string", "null"]},
            },
            "required": [
                "name",
                "sector",
                "industry",
                "country",
                "business_model",
                "key_products_services",
                "revenue_drivers",
                "moat_and_competitive_advantages",
                "cyclicality",
                "capital_intensity",
            ],
        },
        "financial_and_operating_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "top_line_trend": {
                    "type": ["string", "null"],
                    "description": "Narrative on revenue growth/decline, without fabricating exact numbers.",
                },
                "profitability_trend": {
                    "type": ["string", "null"],
                    "description": "Narrative on margins, profitability direction.",
                },
                "balance_sheet_health": {
                    "type": ["string", "null"],
                    "description": "Leverage, liquidity, debt profile qualitatively.",
                },
                "cash_flow_and_capex": {
                    "type": ["string", "null"],
                    "description": "Cash generation, reinvestment needs.",
                },
                "capital_allocation": {
                    "type": ["string", "null"],
                    "description": "Buybacks, dividends, M&A, reinvestment style.",
                },
                "recent_performance_highlights": {
                    "type": ["string", "null"],
                    "description": "Use optional metrics_for_symbol to talk about recent returns and volatility.",
                },
            },
            "required": [
                "top_line_trend",
                "profitability_trend",
                "balance_sheet_health",
                "cash_flow_and_capex",
                "capital_allocation",
                "recent_performance_highlights",
            ],
        },
        "competitive_positioning": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "peer_set": {"type": "array", "items": {"type": "string"}},
                "position_vs_peers": {"type": ["string", "null"]},
                "structural_tailwinds": {"type": "array", "items": {"type": "string"}},
                "structural_headwinds": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "peer_set",
                "position_vs_peers",
                "structural_tailwinds",
                "structural_headwinds",
            ],
        },
        "recent_developments": {
            "type": "array",
            "description": "Stock-specific recent events and news.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "date": {
                        "type": "string",
                        "description": "ISO-8601 or YYYY-MM-DD",
                    },
                    "source": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "summary": {"type": ["string", "null"]},
                    "impact": {"type": ["string", "null"]},
                },
                "required": ["headline", "date", "source", "url", "summary", "impact"],
            },
        },
        "sentiment_snapshot": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_sentiment": {
                    "type": ["string", "null"],
                    "enum": ["bullish", "neutral", "bearish", "mixed", None],
                },
                "drivers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "theme": {"type": "string"},
                            "tone": {
                                "type": ["string", "null"],
                                "enum": ["positive", "neutral", "negative", None],
                            },
                            "commentary": {"type": ["string", "null"]},
                        },
                        "required": ["theme", "tone", "commentary"],
                    },
                },
                "sources_considered": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overall_sentiment", "drivers", "sources_considered"],
        },
        "thesis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bull_case": {"type": ["string", "null"]},
                "base_case": {"type": ["string", "null"]},
                "bear_case": {"type": ["string", "null"]},
                "key_drivers": {"type": "array", "items": {"type": "string"}},
                "typical_time_horizon": {
                    "type": ["string", "null"],
                    "description": "e.g., 'multi-year', '1-3 years', etc.",
                },
            },
            "required": [
                "bull_case",
                "base_case",
                "bear_case",
                "key_drivers",
                "typical_time_horizon",
            ],
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "risk": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "how_to_monitor": {"type": ["string", "null"]},
                },
                "required": ["risk", "why_it_matters", "how_to_monitor"],
            },
        },
        "valuation_context": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relative_positioning": {
                    "type": ["string", "null"],
                    "description": "Qualitative: how the stock tends to trade vs peers/market (growth vs value, premium vs discount), *without* giving explicit price targets.",
                },
                "key_multiples_mentioned": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Narrative references to P/E, EV/EBITDA, etc. If you mention numbers, they must be clearly supported by sources.",
                },
                "valuation_narrative": {
                    "type": ["string", "null"],
                    "description": "Describe how investors generally frame valuation (e.g. 'priced as a high-growth AI leader', 'turnaround story'). Avoid explicit 'cheap'/'expensive' verdicts.",
                },
            },
            "required": ["relative_positioning", "key_multiples_mentioned", "valuation_narrative"],
        },
        "scenarios": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bull": {"type": ["string", "null"]},
                "base": {"type": ["string", "null"]},
                "bear": {"type": ["string", "null"]},
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bull": {"type": ["number", "null"]},
                        "base": {"type": ["number", "null"]},
                        "bear": {"type": ["number", "null"]},
                    },
                    "required": ["bull", "base", "bear"],
                },
            },
            "required": ["bull", "base", "bear", "probabilities"],
        },
        "faq": {
            "type": "array",
            "description": "Optional Q&A style summary for non-expert users.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        },
        "explainability": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "section_confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "company_profile": {"type": ["number", "null"]},
                        "financial_and_operating_summary": {"type": ["number", "null"]},
                        "competitive_positioning": {"type": ["number", "null"]},
                        "recent_developments": {"type": ["number", "null"]},
                        "sentiment_snapshot": {"type": ["number", "null"]},
                        "thesis": {"type": ["number", "null"]},
                        "risks": {"type": ["number", "null"]},
                        "valuation_context": {"type": ["number", "null"]},
                        "scenarios": {"type": ["number", "null"]},
                    },
                    "required": [
                        "company_profile",
                        "financial_and_operating_summary",
                        "competitive_positioning",
                        "recent_developments",
                        "sentiment_snapshot",
                        "thesis",
                        "risks",
                        "valuation_context",
                        "scenarios",
                    ],
                },
                "confidence_overall": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["assumptions", "limitations", "section_confidence", "confidence_overall"],
        },
        "disclaimer": {"type": "string"},
    },
    "required": [
        "symbol",
        "company_profile",
        "financial_and_operating_summary",
        "competitive_positioning",
        "recent_developments",
        "sentiment_snapshot",
        "thesis",
        "risks",
        "valuation_context",
        "scenarios",
        "faq",
        "explainability",
        "disclaimer",
    ],
}
