# services/portfolio_linkup.py
# -*- coding: utf-8 -*-
"""
Linkup orchestration for portfolio AI layers.

- Uses your quotes_map (symbol -> yahoo payload) to extract held symbols.
- Produces ONLY the AI sections: latest_developments, catalysts, scenarios,
  actions, alerts, risks_list, explainability, section_confidence, summary, disclaimer.
- Provides a merge helper to combine AI sections with your precomputed_core
  without overwriting deterministic metrics/allocations/performance/risk.

Dependencies:
- .linkup_config.client  -> your LinkupClient with .search(...)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Iterable

from .linkup_config import client  # your LinkupClient

# --------------------------- AI-only schema -------------------------------- #

AI_LAYERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "latest_developments": {
            "type": "array",
            "minItems": 3,
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string"},
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                    "cause": {"type": "string"},
                    "impact": {"type": "string"},
                    "assets_affected": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "catalysts": {
            "type": "array",
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "date": {"type": "string"},
                    "type": {"type": "string", "description": "earnings, macro, product, vote"},
                    "description": {"type": "string"},
                    "expected_direction": {"type": "string", "enum": ["up", "down", "unclear"]},
                    "magnitude_basis": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "assets_affected": {"type": "array", "items": {"type": "string"}}
                }
            }
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
                    "properties": {
                        "bull": {"type": "number"},
                        "base": {"type": "number"},
                        "bear": {"type": "number"}
                    }
                }
            }
        },
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
                    "category": {"type": "string", "description": "rebalance, alert, tax, hedge, research"}
                }
            }
        },
        "alerts": {
            "type": "array",
            "description": "Thresholds user asked to monitor",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "e.g., SBSI <-10% from 30D high"},
                    "status": {"type": "string", "enum": ["ok", "triggered", "snoozed"]}
                }
            }
        },
        "risks_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "monitor": {"type": "string"},
                    "assets_affected": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "explainability": {
            "type": "object",
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "confidence_overall": {"type": "number", "minimum": 0, "maximum": 1},
                "limitations": {"type": "array", "items": {"type": "string"}}
            }
        },
        "section_confidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scenarios": {"type": "number"},
                "news": {"type": "number"},
                "actions": {"type": "number"}
            }
        },
        "summary": {"type": "string"},
        "disclaimer": {"type": "string"}
    },
    "required": ["summary", "disclaimer"]
}

# ----------------------------- Helpers ------------------------------------- #

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
    targets: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compact, deterministic instruction block for Linkup.
    """
    return {
        "role": "You are a portfolio analyst. Generate ONLY the requested AI sections.",
        "objectives": [
            "Consolidate latest developments and catalysts linked to held symbols.",
            "Draft bull/base/bear scenarios with probabilities and clear basis.",
            "Propose 3–6 prioritized actions with rationale (rebalance, alert, tax, hedge, research).",
            "List explicit portfolio risks with what to monitor.",
            "Provide a concise summary and a standard disclaimer."
        ],
        "constraints": [
            "STRICTLY conform to the provided JSON schema.",
            "Populate `assets_affected` using the provided held symbols only.",
            "Do NOT invent or echo metrics/allocations/performance already computed by the system.",
            "Omit any field you cannot support; do not write 'N/A'.",
            "This is informational, not investment advice."
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
            "targets": targets or {}
        },
        "news_window_iso_utc": {
            "from": news_from_iso,
            "to": news_to_iso
        }
    }

# ---------------------------- Public API ----------------------------------- #

def get_portfolio_ai_layers_from_quotes(
    *,
    quotes_map: Dict[str, Dict[str, Any]],
    base_currency: str = "USD",
    days_of_news: int = 7,
    include_sources: bool = False,
    timeout: int = 60,
    targets: Optional[Dict[str, float]] = None,
    symbols_preferred_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build AI-only portfolio sections using symbols derived from quotes_map.

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

    Returns
    -------
    Dict[str, Any]  (matches AI_LAYERS_SCHEMA)
    """
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=days_of_news)

    # Determine the vocabulary of symbols we want the model to tag
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
        targets=targets
    )

    query = (
        "Portfolio AI layers only.\n\n"
        "Follow the INSTRUCTION block and produce STRICT JSON per the schema.\n\n"
        f"INSTRUCTION:\n{json.dumps(instruction, ensure_ascii=False)}"
    )

    # Single Linkup call; no retry (per your preference)
    response = client.search(
        query=query,
        depth="standard",
        output_type="structured",
        structured_output_schema=json.dumps(AI_LAYERS_SCHEMA),
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
    base_currency: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge AI sections into your precomputed_core, without overwriting deterministic fields.
    The result matches your full portfolio schema contract.
    """
    out = dict(precomputed_core)  # shallow copy

    # Timestamp hygiene (only if you didn't already set them)
    now_iso = datetime.now(timezone.utc).isoformat()
    out.setdefault("as_of", now_iso)
    if base_currency:
        out.setdefault("base_currency", base_currency)
    out.setdefault("timestamps", {})
    out["timestamps"].setdefault("calculated_at", now_iso)

    # Only attach the AI sections we asked for:
    for key in [
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
    ]:
        if key in ai_layers and ai_layers[key] is not None:
            out[key] = ai_layers[key]

    return out


__all__ = [
    "AI_LAYERS_SCHEMA",
    "get_portfolio_ai_layers_from_quotes",
    "assemble_portfolio_report",
]
