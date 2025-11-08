# -*- coding: utf-8 -*-
"""
Linkup orchestration for portfolio AI layers (v2.1, professional prompt).

What’s new vs v2:
- Keeps ALL original keys (backwards compatible).
- Adds two OPTIONAL, presentation-focused fields:
  * rebalance_paths: Aggressive/Balanced/CapitalPreservation (maps to STEP 4)
  * market_outlook: short_term, medium_term, key_opportunities, key_risks (maps to STEP 4/5)
- Instruction builder now follows the 5-STEP brief (TASK, CONTEXT, REFERENCES, EVALUATE, ITERATE)
  while preserving strict JSON schema compliance and “AI-only, do not overwrite deterministic
  metrics” constraints.

Safe rollout:
- professional_mode=True by default (can be disabled).
- include_sources flag continues to control whether Linkup is asked to retrieve citations.

Dependencies:
- .linkup_config.client -> your LinkupClient with .search(...)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .linkup_config import client  # your LinkupClient

# -----------------------------------------------------------------------------
# v2.1 AI-only schema (backward compatible with v2)
# -----------------------------------------------------------------------------

AI_LAYERS_SCHEMA_V2_1: Dict[str, Any] = {
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

        # ---------------- Performance Analysis ----------------
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

        # ---------------- Sentiment Layer ----------------
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

        # ---------------- Predictions Layer ----------------
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
                            "expected_direction": {"type": "string", "enum": ["up", "down", "neutral"]},
                            "expected_change_pct": {"type": "number"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"},
                        },
                        "required": ["symbol", "expected_direction"],
                    },
                },
            },
        },

        # ---------------- NEW (optional): Rebalance Paths (maps to STEP 4) ----------------
        "rebalance_paths": {
            "type": "object",
            "description": "Three profile-based rebalancing paths.",
            "additionalProperties": False,
            "properties": {
                "aggressive_growth": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "balanced_growth": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "capital_preservation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "allocation_notes": {"type": "array", "items": {"type": "string"}},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },

        # ---------------- NEW (optional): Market Outlook (maps to STEP 4/5) ----------------
        "market_outlook": {
            "type": "object",
            "description": "Short and medium term outlook, with opportunities and risks.",
            "additionalProperties": False,
            "properties": {
                "short_term": {"type": "string"},
                "medium_term": {"type": "string"},
                "key_opportunities": {"type": "array", "items": {"type": "string"}},
                "key_risks": {"type": "array", "items": {"type": "string"}},
            },
        },

        # ---------------- Explainability ----------------
        "explainability": {
            "type": "object",
            "properties": {
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "confidence_overall": {"type": "number", "minimum": 0, "maximum": 1},
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
                        "rebalance_paths": {"type": "number"},
                        "market_outlook": {"type": "number"},
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },

        # v1 compatibility
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

# Public alias for compatibility
AI_LAYERS_SCHEMA_V2 = AI_LAYERS_SCHEMA_V2_1
AI_LAYERS_SCHEMA = AI_LAYERS_SCHEMA_V2_1


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


def _build_instruction_professional(
    *,
    base_currency: str,
    symbols: List[str],
    news_from_iso: str,
    news_to_iso: str,
    targets: Optional[Dict[str, float]],
    forecast_window: str,
) -> Dict[str, Any]:
    """
    Professional 5-STEP instruction (maps to your brief) while keeping strict JSON output.
    """
    return {
        "role": "You are a professional portfolio analyst and financial strategist. "
                "Generate ONLY the requested AI sections per the JSON schema provided.",
        "step_1_task": [
            "Classify holdings into Core / Speculative / Hedge (do NOT restate deterministic weights).",
            "Provide portfolio breakdown, diversification check, performance assessment, and rebalance recommendations "
            "as narrative that maps into actions/rebalance_paths without re-computing hard metrics.",
            "Create three rebalancing paths: Aggressive Growth, Balanced Growth, Capital Preservation.",
            "Provide a data-driven, news-aware market outlook informed by recent developments.",
            "End with an optimized rebalance suggestion (within actions) for the next 6–12 months.",
        ],
        "step_2_context": [
            "Assume North American markets (NASDAQ, NYSE, TSX) as reference.",
            "Incorporate recent market trends, earnings, rates, sector rotations where supported.",
            "Tone: professional yet conversational—like a private wealth advisor.",
            "If growth-oriented, favor AI/tech/ETFs in core; flag higher-risk names (biotech, mining, psychedelics) as speculative.",
        ],
        "step_3_references": [
            "Use credible market insights/news. When quoting or summarizing sources, include inline citations "
            "in the exact format: 【source†L#-L#】 and prefer recent sources.",
            "When unavailable, rely on sensible probabilistic reasoning and clearly mark assumptions in explainability.limitations.",
        ],
        "step_4_evaluate": [
            "Check coherence: are narrative recommendations consistent with observed performance and concentration?",
            "Highlight leaders/laggards and notable shifts; assess diversification and concentration risks.",
            "For the rebalance plan, use Trim/Hold/Add style language in actions and within each rebalance path; avoid prescriptive advice.",
        ],
        "step_5_iterate": [
            "Perform one iteration: refine allocations/suggestions for better balance and expected return; "
            "note the refinement in explainability.assumptions or limitations.",
        ],
        "constraints": [
            "STRICTLY conform to the provided JSON schema.",
            "Populate assets_affected using ONLY the provided held symbols.",
            "Do NOT invent or echo deterministic metrics/allocations/performance computed by the system.",
            "Return only fields supported by evidence or sensible probabilistic reasoning; omit unsupported fields.",
            "All content is informational, not investment advice. Avoid liability-inducing phrasing.",
            "Keep tone clear, data-backed, and concise. Use % and $ in numerical statements where applicable.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
            "targets": targets or {},
        },
        "news_window_iso_utc": {"from": news_from_iso, "to": news_to_iso},
        "predictions": {"forecast_window": forecast_window},
        # Map the STEP deliverables to schema sections the model must populate:
        "deliverables_to_schema": {
            "latest_developments": True,
            "catalysts": True,
            "scenarios": True,
            "actions": True,
            "risks_list": True,
            "performance_analysis": True,
            "sentiment": True,
            "predictions": True,
            "rebalance_paths": True,
            "market_outlook": True,
            "summary": True,
            "disclaimer": True,
        },
    }


def _build_instruction_legacy(
    *,
    base_currency: str,
    symbols: List[str],
    news_from_iso: str,
    news_to_iso: str,
    targets: Optional[Dict[str, float]],
    forecast_window: str,
) -> Dict[str, Any]:
    """
    Original deterministic instruction block (v2 style).
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
    professional_mode: bool = True,
) -> Dict[str, Any]:
    """
    Build AI-only portfolio sections using symbols derived from quotes_map (v2.1).

    Parameters
    ----------
    quotes_map : dict
        Your Yahoo bulk payload: {symbol: {...yahoo fields...}}
    base_currency : str
        Contextual currency label for the narrative.
    days_of_news : int
        Lookback window for news/catalysts.
    include_sources : bool
        Whether to ask Linkup to return sources (citations).
    timeout : int
        Linkup call timeout (seconds).
    targets : Optional[dict]
        Optional target weights (e.g., {"Equities": 60, "Bonds": 30, "Cash": 10})
    symbols_preferred_order : Optional[List[str]]
        If your app has canonical symbols (e.g., without -USD suffix), pass them;
        we’ll use this list as the `assets_affected` vocabulary.
    forecast_window : str
        Predictions horizon (e.g., "30D", "90D"). Informational-only.
    professional_mode : bool
        If True, uses the 5-STEP professional instruction. Otherwise falls back to legacy v2.

    Returns
    -------
    Dict[str, Any]  (matches AI_LAYERS_SCHEMA_V2_1)
    """
    now = datetime.now(timezone.utc)
    to_date = now
    from_date = to_date - timedelta(days=days_of_news)

    symbols = (
        [s.strip().upper() for s in (symbols_preferred_order or []) if s] or
        _extract_symbols_from_quotes(quotes_map)
    )

    instruction = (
        _build_instruction_professional(
            base_currency=base_currency,
            symbols=symbols,
            news_from_iso=from_date.date().isoformat(),
            news_to_iso=to_date.date().isoformat(),
            targets=targets,
            forecast_window=forecast_window,
        )
        if professional_mode
        else _build_instruction_legacy(
            base_currency=base_currency,
            symbols=symbols,
            news_from_iso=from_date.date().isoformat(),
            news_to_iso=to_date.date().isoformat(),
            targets=targets,
            forecast_window=forecast_window,
        )
    )

    # The actual prompt that goes to Linkup
    query = (
        "Portfolio AI layers only.\n\n"
        "Follow the INSTRUCTION block and produce STRICT JSON per the schema.\n\n"
        f"INSTRUCTION:\n{json.dumps(instruction, ensure_ascii=False)}"
    )

    response = client.search(
        query=query,
        depth="standard",
        output_type="structured",
        structured_output_schema=json.dumps(AI_LAYERS_SCHEMA_V2_1),
        include_images=False,
        include_sources=include_sources,
        from_date=from_date.date(),
        to_date=to_date.date(),
        timeout=timeout,
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
    The result matches your full portfolio schema contract. v2.1 adds new keys.
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
        # v2.1 additions
        "rebalance_paths",
        "market_outlook",
    ]

    for key in ai_keys:
        if key in ai_layers and ai_layers[key] is not None:
            out[key] = ai_layers[key]

    return out


__all__ = [
    "AI_LAYERS_SCHEMA",
    "AI_LAYERS_SCHEMA_V2",
    "AI_LAYERS_SCHEMA_V2_1",
    "get_portfolio_ai_layers_from_quotes",
    "assemble_portfolio_report",
]
