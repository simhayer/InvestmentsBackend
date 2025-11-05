# -*- coding: utf-8 -*-
"""
Linkup orchestration for portfolio AI layers (v2).

Adds three optional layers on top of v1:
- performance_analysis: AI commentary on deterministic metrics (leaders/laggards/shifts).
- sentiment: aggregated tone & narrative drivers for held symbols.
- predictions: short-horizon directional outlook per asset (probabilistic, informational-only).

Backwards compatible:
- Keeps all original keys (latest_developments, catalysts, scenarios, actions, alerts,
  risks_list, explainability, section_confidence, summary, disclaimer).
- New keys are optional and omitted when unsupported.

This module:
- Builds a compact instruction for Linkup with constraints & objectives.
- Calls Linkup .search() with your structured schema.
- Merges AI layers into precomputed_core without overwriting deterministic metrics.

Dependencies:
- .linkup_config.client  -> your LinkupClient with .search(...)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .linkup_config import client  # your LinkupClient

# -----------------------------------------------------------------------------
# v2 AI-only schema
# -----------------------------------------------------------------------------

AI_LAYERS_SCHEMA_V2: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        # ---------------- News & Events ----------------
        "latest_developments": {
            "type": "array",
            "minItems": 3,
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string", "description": "ISO-8601 UTC or YYYY-MM-DD"},
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                    "cause": {"type": "string"},
                    "impact": {"type": "string"},
                    "assets_affected": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["headline", "date"],
            },
        },

        # ---------------- Catalysts ----------------
        "catalysts": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "date": {"type": "string", "description": "ISO-8601 UTC or YYYY-MM-DD"},
                    "type": {
                        "type": "string",
                        "description": "earnings, macro, product, vote, guidance, regulatory, etc.",
                    },
                    "description": {"type": "string"},
                    "expected_direction": {"type": "string", "enum": ["up", "down", "unclear", "neutral"]},
                    "magnitude_basis": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "assets_affected": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "description"],
            },
        },

        # ---------------- Scenarios ----------------
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

        # ---------------- Actions ----------------
        "actions": {
            "type": "array",
            "description": "Prioritized, concrete next steps",
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
                        "description": "rebalance, alert, tax, hedge, research, liquidity",
                    },
                },
                "required": ["title", "rationale"],
            },
        },

        # ---------------- Alerts ----------------
        "alerts": {
            "type": "array",
            "description": "Thresholds user asked to monitor",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "e.g., SBSI <-10% from 30D high or BTC volatility > X",
                    },
                    "status": {"type": "string", "enum": ["ok", "triggered", "snoozed"]},
                },
                "required": ["condition", "status"],
            },
        },

        # ---------------- Risks ----------------
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

        # ---------------- NEW: Performance Analysis (commentary on deterministic metrics) ----------------
        "performance_analysis": {
            "type": "object",
            "description": "AI commentary on quantitative metrics (leaders/laggards/shifts).",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "leaders": {"type": "array", "items": {"type": "string"}},
                "laggards": {"type": "array", "items": {"type": "string"}},
                "notable_shifts": {"type": "array", "items": {"type": "string"}},
            },
        },

        # ---------------- NEW: Sentiment Layer ----------------
        "sentiment": {
            "type": "object",
            "description": "Aggregated tone and narrative drivers for held assets.",
            "additionalProperties": False,
            "properties": {
                "overall_sentiment": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
                "sources_considered": {"type": "array", "items": {"type": "string"}},
                "drivers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "theme": {"type": "string"},
                            "tone": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "impact": {"type": "string"},
                        },
                        "required": ["theme"],
                    },
                },
                "summary": {"type": "string"},
            },
        },

        # ---------------- NEW: Predictions Layer ----------------
        "predictions": {
            "type": "object",
            "description": "Short-horizon directional outlook (informational-only).",
            "additionalProperties": False,
            "properties": {
                "forecast_window": {"type": "string", "description": "e.g., '30D', '90D'"},
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

        # ---------------- Explainability ----------------
        "explainability": {
            "type": "object",
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "confidence_overall": {"type": "number", "minimum": 0, "maximum": 1},
                # New: optional per-section confidence here
                "section_confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "news": {"type": "number"},
                        "catalysts": {"type": "number"},
                        "actions": {"type": "number"},
                        "sentiment": {"type": "number"},
                        "predictions": {"type": "number"},
                        "scenarios": {"type": "number"},
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },

        # v1 field remains for compatibility; frontend may still read this
        "section_confidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scenarios": {"type": "number"},
                "news": {"type": "number"},
                "actions": {"type": "number"},
            },
        },

        # ---------------- Summary & Disclaimer ----------------
        "summary": {"type": "string"},
        "disclaimer": {"type": "string"},
    },
    "required": ["summary", "disclaimer"],
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _extract_symbols_from_quotes(quotes_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Use keys of quotes_map if they are symbols; otherwise fall back to payload['symbol'].
    Dedupes and preserves insertion order.
    """
    seen = {}
    for k, v in quotes_map.items():
        sym = (k or "").strip().upper()
        if not sym and isinstance(v, dict):
            sym = (v.get("symbol") or "").strip().upper()
        if sym:
            seen.setdefault(sym, True)
    return list(seen.keys())


def _build_instruction(
    *,
    base_currency: str,
    symbols: List[str],
    news_from_iso: str,
    news_to_iso: str,
    targets: Optional[Dict[str, float]] = None,
    forecast_window: str = "30D",
) -> Dict[str, Any]:
    """
    Deterministic instruction block for Linkup v2.
    """
    return {
        "role": "You are a portfolio analyst. Generate ONLY the requested AI sections.",
        "objectives": [
            # v1
            "Consolidate latest developments and catalysts linked to held symbols.",
            "Draft bull/base/bear scenarios with probabilities and clear basis.",
            "Propose 3–6 prioritized actions with rationale (rebalance, alert, tax, hedge, research).",
            "List explicit portfolio risks with what to monitor.",
            "Provide a concise summary and a standard disclaimer.",
            # v2 additions
            "Summarize quantitative performance drivers (leaders, laggards, notable shifts).",
            "Aggregate sentiment across credible sources and identify narrative drivers.",
            f"Provide a short-horizon predictions layer (window: {forecast_window}) with rationale and confidence.",
        ],
        "constraints": [
            "STRICTLY conform to the provided JSON schema.",
            "Populate `assets_affected` using ONLY the provided held symbols.",
            "Do NOT invent or echo deterministic metrics/allocations/performance computed by the system.",
            "Return only fields you can support with evidence or sensible probabilistic reasoning.",
            "Omit fields you cannot support; do NOT write 'N/A'.",
            "All content is informational, not investment advice.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
            "targets": targets or {},
        },
        "news_window_iso_utc": {"from": news_from_iso, "to": news_to_iso},
        "predictions": {"forecast_window": forecast_window},
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_portfolio_ai_layers_from_quotes(
    *,
    quotes_map: Dict[str, Dict[str, Any]],
    base_currency: str = "USD",
    days_of_news: int = 7,
    include_sources: bool = False,
    timeout: int = 60,
    targets: Optional[Dict[str, float]] = None,
    symbols_preferred_order: Optional[List[str]] = None,
    forecast_window: str = "30D",
) -> Dict[str, Any]:
    """
    Build AI-only portfolio sections using symbols derived from quotes_map (v2).

    Parameters
    ----------
    quotes_map : dict
        Your Yahoo bulk payload: {symbol: {...yahoo fields...}}
    base_currency : str
        Contextual currency label for the narrative.
    days_of_news : int
        Lookback window for news/catalysts.
    include_sources : bool
        Whether to include source URLs/names in the AI response.
    timeout : int
        Linkup call timeout (seconds).
    targets : Optional[dict]
        Optional target weights (e.g., {"Equities": 60, "Bonds": 30, "Cash": 10})
    symbols_preferred_order : Optional[List[str]]
        If your app has canonical symbols (e.g., without -USD suffix), pass them;
        we’ll use this list as the `assets_affected` vocabulary.
    forecast_window : str
        Predictions horizon (e.g., "30D", "90D"). Informational-only.

    Returns
    -------
    Dict[str, Any]  (matches AI_LAYERS_SCHEMA_V2)
    """
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=days_of_news)

    symbols = (
        [s.strip().upper() for s in symbols_preferred_order if s]  # preferred wins
        if symbols_preferred_order
        else _extract_symbols_from_quotes(quotes_map)
    )

    instruction = _build_instruction(
        base_currency=base_currency,
        symbols=symbols,
        news_from_iso=from_date.date().isoformat(),
        news_to_iso=to_date.date().isoformat(),
        targets=targets,
        forecast_window=forecast_window,
    )

    query = (
        "Portfolio AI layers only.\n\n"
        "Follow the INSTRUCTION block and produce STRICT JSON per the schema.\n\n"
        f"INSTRUCTION:\n{json.dumps(instruction, ensure_ascii=False)}"
    )

    response = client.search(
        query=query,
        depth="standard",
        output_type="structured",
        structured_output_schema=json.dumps(AI_LAYERS_SCHEMA_V2),
        include_images=False,
        include_sources=include_sources,
        from_date=from_date.date(),
        to_date=to_date.date(),
    )
    return response


def assemble_portfolio_report(
    *,
    precomputed_core: Dict[str, Any],
    ai_layers: Dict[str, Any],
    base_currency: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge AI sections into your precomputed_core, without overwriting deterministic fields.
    The result matches your full portfolio schema contract. v2 adds new keys.
    """
    out = dict(precomputed_core)  # shallow copy

    # Timestamp hygiene (only if you didn't already set them)
    now_iso = datetime.now(timezone.utc).isoformat()
    out.setdefault("as_of", now_iso)
    if base_currency:
        out.setdefault("base_currency", base_currency)
    out.setdefault("timestamps", {})
    out["timestamps"].setdefault("calculated_at", now_iso)

    # Attach legacy + new AI keys, only if present and non-null
    ai_keys = [
        # v1
        "latest_developments",
        "catalysts",
        "scenarios",
        "actions",
        "alerts",
        "risks_list",
        "explainability",
        "section_confidence",
        "summary",
        "disclaimer",
        # v2 additions
        "performance_analysis",
        "sentiment",
        "predictions",
    ]

    for key in ai_keys:
        if key in ai_layers and ai_layers[key] is not None:
            out[key] = ai_layers[key]

    return out


# Public alias for compatibility (v1 name)
AI_LAYERS_SCHEMA = AI_LAYERS_SCHEMA_V2

__all__ = [
    "AI_LAYERS_SCHEMA",
    "AI_LAYERS_SCHEMA_V2",
    "get_portfolio_ai_layers_from_quotes",
    "assemble_portfolio_report",
]
