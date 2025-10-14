from __future__ import annotations
import os, json, datetime as dt
from typing import Dict, List, TypedDict, Optional, Any
from services.helpers.ai.ai_config import _client, PPLX_MODEL, SEARCH_RECENCY

import httpx

# ---- Input/Output types ----
class NewsItem(TypedDict, total=False):
    title: str
    url: str
    snippet: Optional[str]
    published_at: Optional[str]
    source: Optional[str]
    image: Optional[str]

# What the endpoint will return
class PortfolioSummary(TypedDict, total=False):
    summary: str                 # 2–4 sentences: what matters to this portfolio
    highlights: List[str]        # concise bullet points
    risks: List[str]             # concise bullet points
    per_symbol: Dict[str, str]   # short note per symbol (1–2 lines each)
    sentiment: float             # -1..1 overall tone
    sources: List[str]           # top 5 referenced URLs (deduped)

# ---- LLM provider config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PPLX_API_KEY = os.getenv("PPLX_API_KEY", "")

PROVIDER = os.getenv("SUMMARY_PROVIDER", "perplexity")  # "openai" | "perplexity"
OPENAI_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")  # cheap + good for summaries
PPLX_MODEL  = os.getenv("PPLX_MODEL",  "sonar-small-online")  # example

# ---- Helper: trim context to avoid huge payloads ----
def _select_top_articles(news_by_symbol: Dict[str, List[NewsItem]], max_total: int = 40) -> Dict[str, List[NewsItem]]:
    # Flatten, sort by published_at desc, then keep top N and re-group
    def ts(item: NewsItem) -> int:
        try:
            published_at = item.get("published_at", "")
            if published_at is None:
                published_at = ""
            return int(dt.datetime.fromisoformat(published_at.replace("Z","")).timestamp())
        except Exception:
            return 0
    flat: List[tuple[str, NewsItem]] = []
    for sym, items in news_by_symbol.items():
        for it in items or []:
            flat.append((sym, it))
    flat.sort(key=lambda x: ts(x[1]), reverse=True)
    selected = flat[:max_total]
    out: Dict[str, List[NewsItem]] = {}
    for sym, it in selected:
        out.setdefault(sym, []).append(it)
    return out

def _build_prompt(symbols: List[str], news: Dict[str, List[NewsItem]]) -> str:
    # Compact JSON for model context
    compact = {}
    for s, items in news.items():
        compact[s] = [
            {
                "title": it.get("title", "")[:220],
                "snippet": (it.get("snippet") or "")[:300],
                "source": it.get("source", ""),
                "url": it.get("url", ""),
                "published_at": it.get("published_at",""),
            }
            for it in items
        ]
    return (
        "You are an investment assistant. Summarize how this week's news affects the user's portfolio.\n"
        "Return a concise, actionable brief with:\n"
        "1) A 2–4 sentence portfolio-level summary (no fluff).\n"
        "2) 3–6 Highlights (positives, catalysts, business updates).\n"
        "3) 2–5 Risks (macro, regulatory, execution, competition).\n"
        "4) 1–2 lines per symbol (ticker: takeaway).\n"
        "5) Overall sentiment score in [-1,1].\n"
        "Keep it neutral, factual, and avoid speculation. Prefer official sources and high-quality outlets.\n"
        f"Portfolio tickers: {', '.join(symbols)}\n"
        "Recent articles (JSON):\n"
        + json.dumps(compact, ensure_ascii=False)
    )

# ---- Providers ----
async def _call_openai(prompt: str) -> PortfolioSummary:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    # Strict JSON response via response_format
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Respond in JSON with keys: summary, highlights, risks, per_symbol, sentiment, sources."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 600,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        js = r.json()
        raw = js["choices"][0]["message"]["content"]
        out = json.loads(raw)
        return _coerce_summary(out)

async def _call_perplexity(prompt: str) -> PortfolioSummary:
    if not PPLX_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY missing")
    # Perplexity chat style with JSON result target
    body = {
        "model": PPLX_MODEL,
        "messages": [
            {"role": "system", "content": "Respond in JSON with keys: summary, highlights, risks, per_symbol, sentiment, sources."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    headers = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        js = r.json()
        raw = js["choices"][0]["message"]["content"]
        out = json.loads(raw)
        return _coerce_summary(out)

def _coerce_summary(raw: Dict[str, Any]) -> PortfolioSummary:
    # Ensure stable shape & types
    return PortfolioSummary(
        summary = str(raw.get("summary","")).strip(),
        highlights = [str(x).strip() for x in (raw.get("highlights") or [])][:6],
        risks = [str(x).strip() for x in (raw.get("risks") or [])][:6],
        per_symbol = {str(k): str(v) for k, v in (raw.get("per_symbol") or {}).items()},
        sentiment = float(raw.get("sentiment", 0.0)),
        sources = [str(x) for x in (raw.get("sources") or [])][:8],
    )

# ---- Public entry point ----
async def summarize_portfolio_news(
    news_by_symbol: Dict[str, List[NewsItem]],
    *,
    symbols: Optional[List[str]] = None,
) -> PortfolioSummary:
    symbols = symbols or list(news_by_symbol.keys())
    if not symbols:
        return PortfolioSummary(
            summary="No recent articles were found for your portfolio.",
            highlights=[],
            risks=[],
            per_symbol={},
            sentiment=0.0,
            sources=[],
        )
    trimmed = _select_top_articles(news_by_symbol, max_total=40)
    prompt = _build_prompt(symbols, trimmed)
    
    if PROVIDER == "perplexity":
        return await _call_perplexity(prompt)
    return await _call_openai(prompt)
