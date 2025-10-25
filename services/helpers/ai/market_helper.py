# services/ai/summary_service.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from datetime import datetime, timezone
from . import ai_config

Json = Dict[str, Any]

def _build_system_prompt() -> str:
    return (
        "You are a neutral markets analyst for a dashboard.\n"
        "• EXACTLY six sections.\n"
        "• Each is a Markdown H3 heading (### Title) + EXACTLY two sentences.\n"
        "• Focus on WHAT MOVED and POSSIBLE DRIVERS.\n"
        "• No advice or predictions. Avoid sensational tone.\n"
        "• Do NOT include citation brackets like [1]; sources are handled separately.\n"
    )

def _build_user_prompt(
    market: str,
    now_iso: str,
    allowed_tickers: Optional[Iterable[str]] = None,
) -> str:
    allowed = ", ".join(sorted(set(allowed_tickers or [])))
    allow_clause = (
        f"Only mention tickers from this allow-list if you need to reference any: [{allowed}]. "
        "Do not include any other tickers. "
        if allowed
        else "Avoid specific tickers. "
    )

    # A tiny template to strongly shape formatting
    template = (
        "Output format (follow exactly):\n"
        "### Equities\n"
        "Sentence one.\n"
        "Sentence two.\n"
        "### Rates & Bonds\n"
        "Sentence one.\n"
        "Sentence two.\n"
        "### Commodities\n"
        "Sentence one.\n"
        "Sentence two.\n"
        "### FX & Crypto\n"
        "Sentence one.\n"
        "Sentence two.\n"
        "### Macro & Policy\n"
        "Sentence one.\n"
        "Sentence two.\n"
        "### Market Breadth & Flows\n"
        "Sentence one.\n"
        "Sentence two."
    )

    return (
        f"Provide a concise market overview for the {market} market, dated {now_iso} (America/Toronto). "
        "Summarize moves and likely drivers observed in reputable, recent coverage. "
        f"{allow_clause}"
        "Do not give advice or portfolio actions. "
        "Avoid sensational tone. "
        "Keep numbers approximate unless you are very confident. "
        f"{template}"
    )

def _postprocess(text: str) -> str:
    """
    Lightweight guardrail: keep only the first six ### sections and
    ensure each has at most two sentences.
    """
    import re
    sections = re.split(r"(?m)^###\s+", text.strip())
    # sections[0] may have preamble; drop it
    clean = []
    for s in sections[1:7]:  # take at most 6
        lines = s.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = " ".join(lines[1:]).strip()
        # split into sentences and cap at 2
        sentences = re.split(r"(?<=[.!?])\s+", body)
        body2 = " ".join(sentences[:2]).strip()
        clean.append(f"### {title}\n{body2}")
    return "\n\n".join(clean) if clean else text.strip() or ""

def generate_ai_summary(
    market: str = "US stock",
    allowed_tickers: Optional[Iterable[str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Returns {"text": str|None, "meta": {...}}. Never raises to caller.
    Uses Perplexity (OpenAI-compatible) from ai_config.
    """
    try:
        now_dt = now or datetime.now(timezone.utc)
        # Use local wall clock note (Toronto) but keep ISO-8601 UTC to be explicit
        now_iso = now_dt.isoformat()

        resp = ai_config._client.chat.completions.create(
            model=ai_config.PPLX_MODEL,
            temperature=0.2,
            max_tokens=380,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_prompt(market, now_iso, allowed_tickers)},
            ],
            # Perplexity-specific options (adjust names to your ai_config if you’ve wrapped them)
            extra_body={
                "search_recency_filter": getattr(ai_config, "SEARCH_RECENCY", "7d"),
                # "top_k": 5,  # optional
            },
            # You can also pass request_timeout via your client if supported
        )

        text = (resp.choices[0].message.content or "").strip()
        text = _postprocess(text)

        if not text or text.count("\n### ") < 5 and not text.startswith("### "):
            # Weak output; signal “degraded” so the UI can show a subtle badge
            return {
                "text": text or None,
                "meta": {
                    "provider": "perplexity",
                    "model": ai_config.PPLX_MODEL,
                    "recency": getattr(ai_config, "SEARCH_RECENCY", "7d"),
                    "generated_at": now_iso,
                    "quality": "degraded",
                },
            }

        return {
            "text": text,
            "meta": {
                "provider": "perplexity",
                "model": ai_config.PPLX_MODEL,
                "recency": getattr(ai_config, "SEARCH_RECENCY", "7d"),
                "generated_at": now_iso,
                "quality": "ok",
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
