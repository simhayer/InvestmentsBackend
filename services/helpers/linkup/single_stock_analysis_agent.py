"""
single_stock_agent.py

Agent: single-stock deep dive.

Goal:
    Given a ticker (and optional deterministic metrics), produce a
    high-quality, explainable, non-prescriptive deep-dive on one stock.

Guardrails:
    - No 'buy', 'sell', 'hold', 'add', 'trim', 'overweight', 'underweight'.
    - No price targets or specific allocation advice.
    - Prefer omission over hallucination for fundamentals not clearly supported
      by sources or metrics.
    - Use inline citations: 【source†L#-L#】 wherever external info is used.

Expected usage pattern:
    from single_stock_agent import call_link_up_for_single_stock

    result = call_link_up_for_single_stock(
        symbol="AAPL",
        base_currency="CAD",
        metrics_for_symbol=metrics["per_symbol"].get("AAPL"),
        linkup_client=linkup_client,
    )
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from .linkup_config import client


# ============================================================
# 1. JSON Schema for single-stock deep dive
# ============================================================

SINGLE_STOCK_SUMMARY_SCHEMA: Dict[str, Any] = {
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


# ============================================================
# 2. Instruction builder for Linkup / LLM
# ============================================================

def build_single_stock_query(
    symbol: str,
    base_currency: str = "USD",
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return {
        "role": (
            "You are a professional equity research analyst. "
            "You produce a structured, in-depth but accessible profile for a single stock. "
            "You MUST return ONLY valid JSON that matches SINGLE_STOCK_SUMMARY_SCHEMA. "
            "You are not permitted to give personalized investment advice or trading recommendations."
        ),
        "step_1_task": [
            "Explain what this company does, how it makes money, and where it sits in its industry.",
            "Summarize financial and operating trends (revenue, profitability, leverage, cash flows) "
            "based on recent filings and credible sources, without fabricating exact numbers.",
            "Describe recent material events (earnings, guidance changes, major product launches, "
            "regulatory actions, large deals) and their likely relevance for investors.",
            "Summarize current sentiment and typical investment thesis (bull, base, bear).",
            "Outline key risks, structural tailwinds/headwinds, and provide a qualitative valuation context.",
        ],
        "step_2_context": [
            f"Focus on the stock identified by ticker symbol: {symbol}.",
            f"Assume the user's base currency is {base_currency}. Currency matters only for narrative context; "
            "do not compute portfolio-level P&L here.",
            "Treat deterministic metrics in metrics_for_symbol as read-only facts if provided. "
            "You may reference them qualitatively (e.g. 'The stock has been volatile recently'), "
            "but do NOT override or recompute them.",
            "Assume a sophisticated but non-professional audience: avoid jargon where possible, "
            "explain it briefly when unavoidable.",
        ],
        "step_3_references": [
            "Use credible sources: company filings, investor presentations, major financial news outlets, "
            "and reliable market data providers.",
            "Whenever you rely on an external source, include inline citations in the form 【source†L#-L#】 "
            "inside the relevant text field.",
            "If you cannot find solid support for a specific detail (e.g., an exact margin or revenue number), "
            "either speak qualitatively (e.g. 'high', 'moderate', 'declining') or omit the detail. Do NOT guess.",
        ],
        "step_4_evaluate": [
            "Check that your narrative is internally consistent and grounded in cited sources.",
            "Avoid unsupported strong claims such as 'this stock is cheap/expensive' unless backed by clear context; "
            "prefer 'often discussed as richly valued vs peers' with citations.",
            "Ensure you do NOT use words that can be interpreted as direct investment advice: "
            "avoid 'you should buy', 'you should sell', 'this is a buy/hold/sell', 'add', 'trim', or "
            "specific allocation instructions.",
            "If you are uncertain or the data is limited (e.g., for a very small or illiquid name), "
            "lower the relevant section_confidence and mention these limitations explicitly.",
        ],
        "step_5_iterate": [
            "After drafting the full JSON, perform one pass to simplify wording where possible, "
            "and to remove any implied recommendation or prescriptive language.",
            "Make sure the disclaimer clearly states that this is general, informational analysis only, "
            "not personalized advice.",
        ],
        "constraints": [
            "STRICTLY conform to SINGLE_STOCK_SUMMARY_SCHEMA.",
            "Return ONLY JSON, no markdown, no prose outside the JSON.",
            "Do NOT make up detailed numeric fundamentals (revenue, EPS, margins, price targets) "
            "if they are not clearly supported by sources.",
            "It is always acceptable to say 'information not clearly available from public sources' "
            "rather than hallucinate.",
            "All content is informational and educational, not investment, tax, or legal advice.",
        ],
        "stock_context": {
            "symbol": symbol,
            "base_currency": base_currency,
            "metrics_for_symbol": metrics_for_symbol or {},
        },
    }


# ============================================================
# 3. Convenience function to call Linkup
# ============================================================

def call_link_up_for_single_stock(
    symbol: str,
    base_currency: Optional[str] = None,
    metrics_for_symbol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instruction = build_single_stock_query(
        symbol=symbol,
        base_currency=base_currency,
        metrics_for_symbol=metrics_for_symbol,
    )

    try:    
        response = client.search(
                query=json.dumps(instruction),
                depth="standard",
                output_type="structured",
                structured_output_schema=json.dumps(SINGLE_STOCK_SUMMARY_SCHEMA)
            )ß
    except Exception as e:
        # logger.error(f"Error occurred while fetching portfolio AI layers: {e}")
        return {"error": str(e)}

    # Assuming Linkup client already returns Python dict. If it's a string, json.loads it here.
    return response
