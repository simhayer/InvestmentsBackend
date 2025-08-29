# services/helpers/ai/analyze_holding_linkup.py
from __future__ import annotations

import os, json, datetime as dt
from typing import Any, Dict, List
import httpx
from bs4 import BeautifulSoup
from schemas.holding import HoldingInput
from services.linkup_service import linkup_search
from services.yahoo_service import get_full_stock_data 
from utils.common_helpers import safe_float, pct_change, round, parse_json_strict

ARTICLES_KEEP = int(os.getenv("ARTICLES_PER_HOLDING", "3"))
TRUSTED = {
    "reuters.com","apnews.com","bloomberg.com","cnbc.com","ft.com",
    "wsj.com","marketwatch.com","finance.yahoo.com","sec.gov",
}
NEWS_RECENCY_DAYS = int(os.getenv("NEWS_RECENCY_DAYS", "7"))
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")

# ---- Linkup search + fetch ----
def _fetch_text(url: str, max_chars: int = 2000) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0"}) as cx:
            html = cx.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script","style","noscript"]): s.decompose()
        text = " ".join(soup.get_text(" ").split())[:max_chars]
        host = (httpx.URL(url).host or "").replace("www.","")
        return {"url": url, "host": host, "content": text}
    except Exception:
        return {"url": url, "host": "", "content": ""}

def _rank_and_pick(articles: List[Dict[str, Any]], k: int = ARTICLES_KEEP) -> List[Dict[str, Any]]:
    def score(a: Dict[str, Any]) -> int:
        host = a.get("host","")
        auth = 2 if host in TRUSTED else 0
        rec = 1
        return auth*2 + rec
    picked, seen = [], set()
    for a in sorted(articles, key=score, reverse=True):
        host = a.get("host","")
        if host in seen:
            continue
        picked.append(a)
        seen.add(host)
        if len(picked) >= k:
            break
    return picked

def _yahoo_snapshot(symbol: str) -> Dict[str, Any]:
    if get_full_stock_data is None:
        return {}
    try:
        y = get_full_stock_data(symbol)
        if y.get("status") != "ok":
            return {}
        keep = (
            "name","exchange","currency","current_price","previous_close","pe_ratio",
            "forward_pe","price_to_book","dividend_yield","beta","52_week_high","52_week_low",
            "distance_from_52w_high_pct","distance_from_52w_low_pct","earnings_date_utc"
        )
        out = {k: y.get(k) for k in keep}
        for k in ("current_price","previous_close","pe_ratio","forward_pe","price_to_book",
                  "dividend_yield","beta","52_week_high","52_week_low",
                  "distance_from_52w_high_pct","distance_from_52w_low_pct"):
            out[k] = safe_float(out.get(k))
        return out
    except Exception:
        return {}

# ---- Public: analyze_holding_linkup ----
def analyze_holding_linkup(holding: HoldingInput, llm: Any) -> Dict[str, Any]:
    """
    Market-aware per-holding analysis using Linkup news + optional Yahoo snapshot.
    Accepts a Pydantic HoldingInput or a dict-like, returns a JSON dict.
    """
    h = holding
    as_of = dt.datetime.utcnow().isoformat()

    symbol = (h.get("symbol") or "").upper().strip()
    name = h.get("name")
    holding_type = (h.get("type") or "stock").lower().strip()
    quantity = safe_float(h.get("quantity")) or 0.0
    purchase_price = safe_float(h.get("purchase_price"))
    currency = h.get("currency") or "USD"
    institution = h.get("institution") or "N/A"

    snap = _yahoo_snapshot(symbol)
    current_price = snap.get("current_price") or safe_float(h.get("current_price"))
    cost_basis = (purchase_price or 0.0) * quantity
    market_value = (current_price or 0.0) * quantity if current_price is not None else safe_float(h.get("value")) or 0.0
    pnl_abs = (market_value or 0.0) - (cost_basis or 0.0)
    pnl_pct = pct_change(market_value, cost_basis)

    market_context = {
        "current_price": round(current_price, 4),
        "previous_close": round(snap.get("previous_close"), 4),
        "day_change_pct": round(pct_change(snap.get("current_price"), snap.get("previous_close")), 4) if snap else None,
        "52_week_high": round(snap.get("52_week_high"), 4),
        "52_week_low": round(snap.get("52_week_low"), 4),
        "distance_from_52w_high_pct": round(snap.get("distance_from_52w_high_pct"), 4),
        "distance_from_52w_low_pct": round(snap.get("distance_from_52w_low_pct"), 4),
        "pe_ratio": round(snap.get("pe_ratio"), 4),
        "forward_pe": round(snap.get("forward_pe"), 4),
        "price_to_book": round(snap.get("price_to_book"), 4),
        "dividend_yield": round(snap.get("dividend_yield"), 4),
        "beta": round(snap.get("beta"), 4),
        "exchange": snap.get("exchange"),
        "currency": currency,
    }

    company = name or snap.get("name") or symbol
    query = f'({symbol} OR "{company}") (earnings OR guidance OR upgrade OR downgrade OR acquisition OR outlook OR lawsuit)'
    res = linkup_search(query, limit=ARTICLES_KEEP, freshness_days=NEWS_RECENCY_DAYS)
    items = res["items"]

    fetched: List[Dict[str, Any]] = []
    for it in items:
        url = it.get("url")
        if not url:
            continue
        page = _fetch_text(url)  # returns {"url","host","excerpt"}
        page["title"] = it.get("title") or url
        page["published_at"] = it.get("published_at")
        # If Linkup already gave a snippet, keep it as a backup excerpt
        snip = it.get("snippet")
        if isinstance(snip, str) and snip and not page.get("excerpt"):
            page["excerpt"] = snip[:800]
        fetched.append(page)

    # Now rank/dedupe your fetched pages and keep the top N
    articles = _rank_and_pick(fetched, k=ARTICLES_KEEP)

    style_hint = {
        "stock": "Focus on valuation (P/E, yield), 52-week range, and near-term earnings.",
        "etf": "Focus on sector/country concentration and 52-week context.",
        "crypto": "Emphasize volatility and position sizing.",
    }.get(holding_type, "Focus on the most relevant risks for this instrument.")

    payload = {
        "holding": {
            "symbol": symbol, "name": company, "type": holding_type,
            "quantity": quantity, "purchase_price": purchase_price,
            "institution": institution, "currency": currency,
        },
        "computed": {
            "cost_basis": round(cost_basis, 4),
            "market_value": round(market_value, 4),
            "pnl_abs": round(pnl_abs, 4),
            "pnl_pct": round(pnl_pct, 4),
        },
        "market_context": market_context,
        "articles": [
            {"title": a.get("title"), "url": a.get("url"), "host": a.get("host"),
             "published_at": a.get("published_at"), "excerpt": a.get("content")}
            for a in articles
        ],
        "style_hint": style_hint,
        "must_follow": [
            "Use ONLY numbers and URLs provided here.",
            "No price targets or personalized financial advice.",
            "If earnings_date_utc is within 14 days, mention event risk.",
            "If distance_from_52w_high_pct ≤ 5 or distance_from_52w_low_pct ≤ 5, call that out.",
            "Return VALID JSON only with the schema below.",
        ],
        "output_schema": {
            "symbol": "string",
            "as_of_utc": "ISO-8601 string",
            "pnl_abs": "number",
            "pnl_pct": "number | null",
            "market_context": "object (as provided)",
            "rating": "one of: hold | sell | watch | diversify",
            "rationale": "short paragraph grounded in provided facts/links",
            "key_risks": "array of 3-5 short bullets",
            "suggestions": "array of 2-4 actionable, non-prescriptive tips with optional citations",
            "data_notes": "array of short strings about data/provenance",
            "disclaimer": "string: 'This is educational information, not financial advice.'",
        },
        "disclaimer": "This is educational information, not financial advice.",
    }

    system = (
        "You are a cautious retail investing explainer. "
        "Use ONLY the provided fields and article URLs; do not invent numbers or dates. "
        "Be concise and specific. Return VALID JSON ONLY per the schema."
    )

    ai = llm.invoke([
        {"role":"system","content": system},
        {"role":"user","content": json.dumps(payload)},
    ])

    try:
        obj = parse_json_strict(ai.content)
    except Exception:
        obj = {
            "symbol": symbol,
            "as_of_utc": as_of,
            "pnl_abs": round(pnl_abs, 4),
            "pnl_pct": round(pnl_pct, 4),
            "market_context": market_context,
            "rating": "watch",
            "rationale": "Automatic fallback: could not parse model output. Based on provided snapshot/news, monitor updates.",
            "key_risks": [],
            "suggestions": [],
            "data_notes": [
                f"News via Linkup (last {NEWS_RECENCY_DAYS} days).",
                "Snapshot source: Yahoo (if available).",
            ],
            "disclaimer": "This is educational information, not financial advice.",
        }

    # Enforce trusted fields & defaults
    obj.setdefault("symbol", symbol)
    obj.setdefault("as_of_utc", as_of)
    obj.setdefault("pnl_abs", round(pnl_abs, 4))
    obj.setdefault("pnl_pct", round(pnl_pct, 4))
    obj["market_context"] = market_context
    dn = obj.get("data_notes") or []
    if not LINKUP_API_KEY:
        dn.append("Linkup API key missing; news step skipped.")
    if get_full_stock_data is None:
        dn.append("Yahoo snapshot disabled; using holding snapshot only.")
    if dn:
        obj["data_notes"] = dn
    obj.setdefault("disclaimer", "This is educational information, not financial advice.")
    valid = {"hold","sell","watch","diversify"}
    r = str(obj.get("rating","watch")).lower().strip()
    obj["rating"] = r if r in valid else "watch"

    return obj

# we are not using llm here, returning linkup answer and sources
def analyze_holding_only_linkup(holding: HoldingInput) -> Any:
    symbol = (holding.get("symbol") or "").upper().strip()
    name = holding.get("name")

    query = f'({symbol} OR "{name}") (earnings OR guidance OR upgrade OR downgrade OR acquisition OR outlook OR lawsuit)'
    return linkup_search(query, limit=ARTICLES_KEEP, freshness_days=NEWS_RECENCY_DAYS)
    