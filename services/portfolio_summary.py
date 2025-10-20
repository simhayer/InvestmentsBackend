# services/portfolio_summary.py
from __future__ import annotations
from typing import Any, Dict, List

from fastapi import HTTPException
from services.helpers.ai.ai_config import _client, PPLX_MODEL, SEARCH_RECENCY  # same as your analyzer_ai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

MAX_NEWS_PER_SYMBOL = 10
MAX_TITLE_CHARS = 220
MAX_SNIPPET_CHARS = 500
MAX_TOTAL_CHARS = 12000  # keep payload sane to avoid 400

def _trim(s: str | None, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else (s[:n] + "…")

def _build_news_prompt(news_by_symbol: Dict[str, List[Dict[str, Any]]], symbols: List[str]) -> List[ChatCompletionMessageParam]:
    lines: List[str] = [
        "You are an equity/portfolio analyst. Summarize cross-asset news with portfolio impact.",
        "- Prioritize items within the last 7–14 days.",
        "- Group by symbol; highlight risks, catalysts, and whether to up/downsize positions."
    ]
    size = 0
    for sym in symbols:
        items = (news_by_symbol.get(sym, []) or [])[:MAX_NEWS_PER_SYMBOL]
        if not items:
            continue
        lines.append(f"\n### {sym}")
        for nitem in items:
            title = _trim(str(nitem.get("title") or nitem.get("headline") or ""), MAX_TITLE_CHARS)
            src   = nitem.get("source") or "Unknown"
            ts    = nitem.get("published_at") or nitem.get("date") or ""
            gist  = _trim(str(nitem.get("summary") or nitem.get("snippet") or ""), MAX_SNIPPET_CHARS)
            chunk = f"- ({ts}) [{src}] {title}\n  {gist}"
            lines.append(chunk)
            size += len(chunk)
            if size >= MAX_TOTAL_CHARS:
                lines.append("\n...[truncated for length]")
                break
        if size >= MAX_TOTAL_CHARS:
            break

    user_text = "\n".join(lines)

    sys_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": "..." }
    usr_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user_text}
    return [sys_msg, usr_msg]


async def summarize_portfolio_news(news_by_symbol: Dict[str, List[Dict[str, Any]]], *, symbols: List[str]) -> str:
    if not symbols:
        return "No symbols provided."
    messages = _build_news_prompt(news_by_symbol, symbols)

    try:
        resp = _client.chat.completions.create(
            model=PPLX_MODEL,                  # e.g., "sonar" / "sonar-pro" / "sonar-reasoning"
            temperature=0.2,
            max_tokens=900,
            messages=messages,
            # Use the same extras you already use elsewhere:
            extra_body={"search_recency_filter": SEARCH_RECENCY, "return_images": False},
        )
        text = (resp.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        # Surface a clean 4xx/5xx upstream instead of 500 mystery
        raise HTTPException(status_code=502, detail=f"Perplexity request failed: {e}")
