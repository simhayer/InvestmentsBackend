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
                "sector": {"type": "string"},
                "industry": {"type": "string"},
                "country": {"type": "string"},
                "business_model": {"type": "string"},
                "key_products_services": {"type": "array", "items": {"type": "string"}},
                "revenue_drivers": {"type": "array", "items": {"type": "string"}},
                "moat_and_competitive_advantages": {"type": "string"},
                "cyclicality": {"type": "string"},
                "capital_intensity": {"type": "string"},
            },
        },
        "financial_and_operating_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "top_line_trend": {
                    "type": "string",
                    "description": "Narrative on revenue growth/decline, without fabricating exact numbers.",
                },
                "profitability_trend": {
                    "type": "string",
                    "description": "Narrative on margins, profitability direction.",
                },
                "balance_sheet_health": {
                    "type": "string",
                    "description": "Leverage, liquidity, debt profile qualitatively.",
                },
                "cash_flow_and_capex": {
                    "type": "string",
                    "description": "Cash generation, reinvestment needs.",
                },
                "capital_allocation": {
                    "type": "string",
                    "description": "Buybacks, dividends, M&A, reinvestment style.",
                },
                "recent_performance_highlights": {
                    "type": "string",
                    "description": "Use optional metrics_for_symbol to talk about recent returns and volatility.",
                },
            },
        },
        "competitive_positioning": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "peer_set": {"type": "array", "items": {"type": "string"}},
                "position_vs_peers": {"type": "string"},
                "structural_tailwinds": {"type": "array", "items": {"type": "string"}},
                "structural_headwinds": {"type": "array", "items": {"type": "string"}},
            },
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
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                    "summary": {"type": "string"},
                    "impact": {"type": "string"},
                },
                "required": ["headline", "date"],
            },
        },
        "sentiment_snapshot": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_sentiment": {
                    "type": "string",
                    "enum": ["bullish", "neutral", "bearish", "mixed"],
                },
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
                            "commentary": {"type": "string"},
                        },
                        "required": ["theme"],
                    },
                },
                "sources_considered": {"type": "array", "items": {"type": "string"}},
            },
        },
        "thesis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bull_case": {"type": "string"},
                "base_case": {"type": "string"},
                "bear_case": {"type": "string"},
                "key_drivers": {"type": "array", "items": {"type": "string"}},
                "typical_time_horizon": {
                    "type": "string",
                    "description": "e.g., 'multi-year', '1-3 years', etc.",
                },
            },
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "risk": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "how_to_monitor": {"type": "string"},
                },
                "required": ["risk", "why_it_matters"],
            },
        },
        "valuation_context": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relative_positioning": {
                    "type": "string",
                    "description": "Qualitative: how the stock tends to trade vs peers/market (growth vs value, premium vs discount), *without* giving explicit price targets.",
                },
                "key_multiples_mentioned": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Narrative references to P/E, EV/EBITDA, etc. If you mention numbers, they must be clearly supported by sources.",
                },
                "valuation_narrative": {
                    "type": "string",
                    "description": "Describe how investors generally frame valuation (e.g. 'priced as a high-growth AI leader', 'turnaround story'). Avoid explicit 'cheap'/'expensive' verdicts.",
                },
            },
        },
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
                        "company_profile": {"type": "number"},
                        "financial_and_operating_summary": {"type": "number"},
                        "competitive_positioning": {"type": "number"},
                        "recent_developments": {"type": "number"},
                        "sentiment_snapshot": {"type": "number"},
                        "thesis": {"type": "number"},
                        "risks": {"type": "number"},
                        "valuation_context": {"type": "number"},
                        "scenarios": {"type": "number"},
                    },
                },
                "confidence_overall": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
        "disclaimer": {"type": "string"},
    },
    "required": ["symbol", "company_profile", "disclaimer"],
}