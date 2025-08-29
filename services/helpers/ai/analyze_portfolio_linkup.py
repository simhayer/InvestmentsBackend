# services/analysis_entrypoint_linkup.py
from __future__ import annotations

import os
import json
import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI

# Optional: if you want a Yahoo snapshot (52w, P/E, etc.)
try:
    from services.yahoo_service import get_full_stock_data  # pragma: no cover
except Exception:  # pragma: no cover
    get_full_stock_data = None  # type: ignore

# -----------------------------
# Config
# -----------------------------
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
LINKUP_API_URL = os.getenv("LINKUP_API_URL", "https://api.linkup.dev/v1/search")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # OpenAI-compatible
NEWS_RECENCY_DAYS = int(os.getenv("NEWS_RECENCY_DAYS", "7"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "8"))
ARTICLES_PER_SYMBOL = int(os.getenv("ARTICLES_PER_SYMBOL", "3"))

TRUSTED = {
    "reuters.com", "apnews.com", "bloomberg.com", "cnbc.com", "ft.com",
    "wsj.com", "marketwatch.com", "finance.yahoo.com", "sec.gov",
}

# -----------------------------
# Utilities (math & parsing)
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _round(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), d)

def _position_value(h: Dict[str, Any]) -> float:
    v = _safe_float(h.get("value"))
    if v is not None:
        return v
    q = _safe_float(h.get("quantity")) or 0.0
    cp = _safe_float(h.get("current_price")) or 0.0
    return q * cp

def _cost_basis(h: Dict[str, Any]) -> float:
    q = _safe_float(h.get("quantity")) or 0.0
    pp = _safe_float(h.get("purchase_price")) or 0.0
    return q * pp

def _pct_change(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base in (None, 0) or (isinstance(base, float) and abs(base) < 1e-6):
        return None
    return (cur / base - 1.0) * 100.0

def _exposures(positions: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    total_mv = sum(p["market_value"] for p in positions)
    bucket: Dict[str, float] = {}
    for p in positions:
        k = str(p.get(key) or "N/A")
        bucket[k] = bucket.get(k, 0.0) + p["market_value"]
    items = [{"label": k, "weight_pct": (v / total_mv * 100.0) if total_mv > 0 else 0.0} for k, v in bucket.items()]
    items.sort(key=lambda x: x["weight_pct"], reverse=True)
    if key == "type":
        return [{"type": i["label"], "weight_pct": _round(i["weight_pct"], 4)} for i in items]
    if key == "currency":
        return [{"currency": i["label"], "weight_pct": _round(i["weight_pct"], 4)} for i in items]
    return [{"institution": i["label"], "weight_pct": _round(i["weight_pct"], 4)} for i in items]

def _hhi(weights: List[float]) -> float:
    f = [w / 100.0 for w in weights]
    return float(sum(w*w for w in f))

def _parse_json_strict(maybe: Any) -> Dict[str, Any]:
    import re
    if isinstance(maybe, list):
        maybe = "".join(str(p) for p in maybe if p is not None)
    if maybe is None:
        raise ValueError("Empty LLM response (None)")
    if not isinstance(maybe, str):
        maybe = str(maybe)
    s = maybe.strip()
    if not s:
        raise ValueError("Empty LLM response (blank)")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", s)
        if not m:
            raise
        return json.loads(m.group(0))

# -----------------------------
# Linkup search + fetch
# -----------------------------
def _linkup_search(query: str, limit: int = SEARCH_LIMIT, freshness_days: int = NEWS_RECENCY_DAYS) -> List[Dict[str, Any]]:
    if not LINKUP_API_KEY:
        return []
    params = {"q": query, "limit": limit, "freshness": f"{freshness_days}d"}
    headers = {"Authorization": f"Bearer {LINKUP_API_KEY}"}
    try:
        with httpx.Client(timeout=10.0) as cx:
            r = cx.get(LINKUP_API_URL, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])
    except Exception:
        return []

def _fetch_text(url: str, max_chars: int = 2000) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0"}) as cx:
            html = cx.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script","style","noscript"]):
            s.decompose()
        text = " ".join(soup.get_text(" ").split())[:max_chars]
        host = httpx.URL(url).host or ""
        return {"url": url, "host": host, "content": text}
    except Exception:
        return {"url": url, "host": "", "content": ""}

def _rank_and_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    picked, seen_hosts = [], set()
    def score(it: Dict[str, Any]) -> int:
        host = (httpx.URL(it.get("url","")).host or "").replace("www.","")
        auth = 2 if host in TRUSTED else 0
        rec = 1  # Linkup already sorted by freshness; keep simple
        return auth*2 + rec
    for it in sorted(items, key=score, reverse=True):
        host = (httpx.URL(it.get("url","")).host or "")
        if host in seen_hosts:
            continue
        picked.append(it)
        seen_hosts.add(host)
        if len(picked) >= ARTICLES_PER_SYMBOL:
            break
    return picked

# -----------------------------
# LLM (OpenAI-compatible) setup
# -----------------------------
llm = ChatOpenAI(
    temperature=0.2,
    model=LLM_MODEL,
    max_completion_tokens=900,
    timeout=30,
    model_kwargs={"response_format": {"type": "json_object"}},  # JSON mode
)

# -----------------------------
# Public API — HOLDING (kept same signature)
# -----------------------------
from schemas.holding import HoldingInput
from schemas.ai_analysis import AnalysisOutput
from services.helpers.ai.analyze_holding import analyze_holding  # reuse your deterministic single-holding

def analyze_investment_holding(holding: HoldingInput) -> AnalysisOutput:
    # Keep your per-holding path as-is (it already uses Yahoo + JSON LLM explanation)
    return analyze_holding(holding, llm)

# -----------------------------
# Public API — PORTFOLIO (Linkup + OpenAI)
# -----------------------------
def analyze_portfolio_linkup(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Portfolio-level analysis with live news via Linkup + JSON-mode OpenAI.
    Preserves the shape your frontend expects (totals/exposures/concentration/etc.)
    and adds grounded suggestions with citations.
    """
    as_of = dt.datetime.utcnow().isoformat()

    # Aggregate core numbers
    positions: List[Dict[str, Any]] = []
    total_mv, total_cb = 0.0, 0.0
    for h in holdings:
        mv = _position_value(h)
        cb = _cost_basis(h)
        total_mv += mv
        total_cb += cb
        positions.append({
            "symbol": (h.get("symbol") or "").upper(),
            "name": h.get("name"),
            "type": (h.get("type") or "equity").lower(),
            "currency": h.get("currency") or "USD",
            "institution": h.get("institution") or "N/A",
            "market_value": mv,
            "cost_basis": cb,
            "pnl_abs": mv - cb,
            "pnl_pct": _pct_change(mv, cb),
        })

    if total_mv <= 0:
        return {"error": "No funded positions to analyze", "as_of_utc": as_of}

    # Weights, top positions
    for p in positions:
        p["weight_pct"] = (p["market_value"] / total_mv * 100.0) if total_mv > 0 else 0.0
    positions.sort(key=lambda x: x["weight_pct"], reverse=True)
    top_positions = [{"symbol": p["symbol"], "weight_pct": _round(p["weight_pct"], 4), "pnl_pct": _round(p["pnl_pct"], 4)} for p in positions[:5]]

    # Concentration & diversification score
    weights = [p["weight_pct"] for p in positions]
    hh_index = _hhi(weights)
    top1 = weights[0] if weights else 0.0
    top3 = sum(weights[:3]) if weights else 0.0
    top5 = sum(weights[:5]) if weights else 0.0

    hhi_penalty = hh_index * 100.0
    top3_penalty = max(0.0, top3 - 40.0) * 0.5
    currency_exposure = _exposures(positions, "currency")
    currency_top = currency_exposure[0]["weight_pct"] if currency_exposure else 0.0
    currency_penalty = max(0.0, currency_top - 80.0) * 0.5
    diversification_score = max(0.0, 100.0 - hhi_penalty - top3_penalty - currency_penalty)

    # Exposures
    by_type = _exposures(positions, "type")
    by_currency = currency_exposure
    by_institution = _exposures(positions, "institution")

    # Pick top symbols and gather facts via Linkup
    top_syms = [p["symbol"] for p in positions[:3] if p["symbol"]]
    articles_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    for sym in top_syms:
        company = next((p["name"] for p in positions if p["symbol"] == sym and p.get("name")), sym)
        query = f'({sym} OR "{company}") (earnings OR guidance OR upgrade OR downgrade OR acquisition OR outlook OR lawsuit)'
        hits = _linkup_search(query, limit=SEARCH_LIMIT, freshness_days=NEWS_RECENCY_DAYS)
        fetched: List[Dict[str, Any]] = []
        for h in hits:
            url = h.get("url")
            if not url:
                continue
            page = _fetch_text(url)
            page["title"] = h.get("title") or h.get("name") or url
            page["published_at"] = h.get("published_at")
            fetched.append(page)
        articles_by_symbol[sym] = _rank_and_dedupe(fetched)

    # Optional: Yahoo snapshot per symbol (52w, P/E, etc.)
    snapshots: Dict[str, Dict[str, Any]] = {}
    if get_full_stock_data is not None:
        for sym in top_syms:
            try:
                y = get_full_stock_data(sym)  # your service
                if y.get("status") == "ok":
                    keep = (
                        "name","exchange","currency","current_price","previous_close","pe_ratio",
                        "forward_pe","dividend_yield","beta","52_week_high","52_week_low",
                        "distance_from_52w_high_pct","distance_from_52w_low_pct","earnings_date_utc",
                    )
                    snapshots[sym] = {k: y.get(k) for k in keep}
                else:
                    snapshots[sym] = {}
            except Exception:
                snapshots[sym] = {}
    else:
        snapshots = {sym: {} for sym in top_syms}

    # Build payload for the LLM (facts only)
    payload = {
        "as_of_utc": as_of,
        "portfolio": {
            "market_value": _round(total_mv, 2),
            "cost_basis": _round(total_cb, 2),
            "pnl_abs": _round(total_mv - total_cb, 2),
            "pnl_pct": _round(_pct_change(total_mv, total_cb), 2),
            "usd_weight_pct": _round(next((x["weight_pct"] for x in by_currency if (x.get("currency") or "").upper()=="USD"), 0.0), 2),
            "positions_count": len([p for p in positions if p["market_value"] > 0]),
        },
        "top": [
            {
                "symbol": sym,
                "snapshot": snapshots.get(sym, {}),
                "articles": [
                    {"title": a.get("title"), "url": a.get("url"), "host": a.get("host"),
                     "published_at": a.get("published_at"), "excerpt": a.get("content")}
                    for a in (articles_by_symbol.get(sym) or [])
                ],
            }
            for sym in top_syms
        ],
        "must_follow": [
            "Use ONLY the facts and URLs provided here.",
            "No price targets. No personalized financial advice.",
            "Return VALID JSON with: rating(constructive|cautious|defensive), risk_level(low|moderate|high), "
            "rationale (1–3 sentences), highlights[], suggestions[{text,citations?:[url],applies_to?:string}], disclaimer.",
            "If earnings are within 14 days for a top symbol, mention event risk.",
            "If a top symbol is within 5% of its 52-week high or low (from snapshot), call it out.",
        ],
    }

    # Ask the model to write the MARKET-AWARE brief (JSON)
    messages = [
        {"role": "system", "content": "You are a market analyst. Ground everything in the provided facts and URLs. Respond as VALID JSON."},
        {"role": "user", "content": json.dumps(payload)},
    ]
    ai = llm.invoke(messages)

    # Parse robustly
    try:
        ai_obj = _parse_json_strict(ai.content)
    except Exception as e:
        ai_obj = {
            "rating": "cautious",
            "risk_level": "moderate",
            "rationale": f"Could not parse LLM output: {type(e).__name__}. Using computed metrics only.",
            "highlights": [],
            "suggestions": [
                {"text": "Retry analysis in a moment; if it persists, reduce top symbols to 2."}
            ],
            "disclaimer": "This is educational information, not financial advice.",
        }

    # Sanity for required fields
    for k, v in {
        "rating": "cautious",
        "risk_level": "moderate",
        "rationale": "Summary not available.",
        "highlights": [],
        "suggestions": [],
        "disclaimer": "This is educational information, not financial advice.",
    }.items():
        ai_obj.setdefault(k, v)

    # Final response in the shape your UI expects
    return {
        "as_of_utc": as_of,
        "totals": {
            "market_value": _round(total_mv, 2),
            "cost_basis": _round(total_cb, 2),
            "pnl_abs": _round(total_mv - total_cb, 2),
            "pnl_pct": _round(_pct_change(total_mv, total_cb), 2),
        },
        "exposures": {
            "by_type": by_type,
            "by_currency": by_currency,
            "by_institution": by_institution,
        },
        "top_positions": top_positions,
        "concentration": {
            "hh_index": _round(hh_index, 4),
            "top_1_weight_pct": _round(top1, 2),
            "top_3_weight_pct": _round(top3, 2),
            "top_5_weight_pct": _round(top5, 2),
        },
        "diversification_score": _round(diversification_score, 1),
        "rating": ai_obj.get("rating"),
        "risk_level": ai_obj.get("risk_level"),
        "rationale": ai_obj.get("rationale"),
        "suggestions": ai_obj.get("suggestions"),
        "data_notes": [
            f"News fetched via Linkup (last {NEWS_RECENCY_DAYS} days).",
            f"Articles per top symbol: {ARTICLES_PER_SYMBOL}.",
            "Numbers (totals/exposures) computed from stored holdings snapshot.",
        ],
        "disclaimer": ai_obj.get("disclaimer"),
    }
