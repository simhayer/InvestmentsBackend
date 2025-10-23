# services/ai/summary_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from . import ai_config  # your provided config

Json = Dict[str, Any]

def _fmt_item(i: Json) -> str:
    key = i.get("key") or i.get("symbol")
    label = i.get("label") or key
    px = i.get("price")
    dabs = i.get("changeAbs")
    dpct = i.get("changePct")
    ccy = i.get("currency") or "USD"

    if px is None:
        return f"- {label}: N/A (error={i.get('error')})"
    # keep it compact for token-efficiency
    pct = f"{dpct:.2f}%" if isinstance(dpct, (int, float)) else "—"
    abs_ = f"{dabs:+.2f}" if isinstance(dabs, (int, float)) else "—"
    return f"- {label}: {px:.2f} {ccy}  ({abs_}, {pct})"

def _build_system_prompt() -> str:
    return (
        "You are an analyst generating a crisp, neutral market overview for a dashboard. "
        "Be accurate, brief (60–120 words), and focus on **what moved** and **possible drivers**. "
        "Avoid advice; no tickers beyond the provided list; avoid sensational tone."
    )

def _build_user_prompt(items: List[Json]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = "\n".join(_fmt_item(i) for i in items)
    return (
        f"Now: {now}\n"
        "Indices & BTC snapshot:\n"
        f"{lines}\n\n"
        "Write 2–4 short sentences:\n"
        "1) One-line big picture (risk-on/off, breadth if clear).\n"
        "2) Note notable movers (e.g., tech-heavy vs cyclicals signal) without new tickers.\n"
        "3) Briefly mention macro catalysts only if obvious (rates, earnings season, crypto flow)."
    )

def generate_ai_summary(items: List[Json]) -> Dict[str, Any]:
    """
    Returns {"text": str|None, "meta": {...}}. Never raises to caller.
    Uses Perplexity (OpenAI-compatible) from ai_config.
    """
    try:
        # OpenAI-compatible client pointed at Perplexity
        resp = ai_config._client.chat.completions.create(
            model=ai_config.PPLX_MODEL,
            temperature=0.2,
            max_tokens=220,
            messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(items)},
        ],
            # Perplexity-specific search recency
            extra_body={"search_recency": ai_config.SEARCH_RECENCY},
        )

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return {"text": None, "meta": {"provider": "perplexity", "model": ai_config.PPLX_MODEL, "error": "empty"}}

        return {
            "text": text,
            "meta": {
                "provider": "perplexity",
                "model": ai_config.PPLX_MODEL,
                "recency": ai_config.SEARCH_RECENCY,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    except Exception as e:
        return {
            "text": None,
            "meta": {
                "provider": "perplexity",
                "model": getattr(ai_config, "PPLX_MODEL", "unknown"),
                "error": str(e),
            },
        }
