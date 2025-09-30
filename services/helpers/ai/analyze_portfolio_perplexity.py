# services/portfolio/analyzer_ai.py
from __future__ import annotations

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import json
from datetime import datetime, timezone

# ---- AI client & JSON helpers (adjust paths to your project) ----
from services.helpers.ai.json_helpers import S, E, extract_json
from services.helpers.ai.ai_config import _client, PPLX_MODEL, SEARCH_RECENCY  # you already use these

# ---------- Public API ----------

@dataclass
class Position:
    symbol: str
    weight: float                 # normalized [0,1]
    sector: Optional[str] = None
    country: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None  # optional per-symbol analysis blob


def analyze_portfolio_pplx(
    rows: List[Dict[str, Any]],
    *,
    symbol_analyzer: Optional[Callable[[str], Dict[str, Any]]] = None,  # your existing per-symbol function
    analyze_top_n: int = 12,
    min_weight_for_analysis: float = 0.02,
    max_workers: int = 6,
    portfolio_return_pct: Optional[float] = None,
    benchmark_return_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fully AI-driven classifications & insights. Output shape compatible with your hook:
    { summary, positions, insights, as_of_utc?, narrative_markdown? }
    """
    # 1) Collapse by symbol and compute absolute values
    totals = _totals_by_symbol(rows)                          # symbol -> total_value

    # 2) Build normalized positions (no sector/country yet)
    positions = _positions_from_totals(totals)

    # 3) Optional per-symbol deep analyses (your function)
    if symbol_analyzer and positions:
        to_analyze = sorted(positions, key=lambda p: p.weight, reverse=True)
        if analyze_top_n is not None:
            to_analyze = to_analyze[:max(0, analyze_top_n)]
        to_analyze = [p for p in to_analyze if p.weight >= max(0.0, float(min_weight_for_analysis))]
        if to_analyze:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_safe_analyze, symbol_analyzer, p.symbol): p for p in to_analyze}
                for fut in as_completed(futures):
                    p = futures[fut]
                    p.analysis = fut.result()

    # 4) AI classifications (sector / country / asset class, etc.)
    sym_list = [p.symbol for p in positions]
    class_map = classify_symbols_pplx(sym_list)   # {sym: {...}}
    for p in positions:
        info = class_map.get(p.symbol, {})
        # prefer AI classifications; fallback remains None if missing
        p.sector = _clean_or(p.sector, info.get("sector"))
        p.country = _clean_or(p.country, info.get("country"))

    # 5) Aggregate summary stats (now that sector/country are filled by AI)
    summary = _summary(positions)

    # 6) AI insights (no hardcoded rules)
    insights, narrative_md = generate_portfolio_insights_pplx(
        summary=summary,
        positions=[{"symbol": p.symbol, "weight": p.weight, "sector": p.sector, "country": p.country, "analysis": p.analysis} for p in positions],
        rows=rows,
        portfolio_return_pct=portfolio_return_pct,
        benchmark_return_pct=benchmark_return_pct,
    )

    # 7) Assemble response
    return {
        "as_of_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "positions_count": len(positions),
            "by_sector": summary["by_sector"],
            "by_country": summary["by_country"],
            "hhi": summary["hhi"],
            "top_position": {
                "symbol": summary["top_position"][0],
                "weight": summary["top_position"][1],
            },
        },
        "positions": [
            {
                "symbol": p.symbol,
                "weight": p.weight,
                "sector": p.sector,
                "country": p.country,
                "analysis": p.analysis,
            } for p in positions
        ],
        "insights": insights,
        "narrative_markdown": narrative_md,  # optional rich summary for your UI
    }

# ---------- AI: symbol classification ----------

CLASSIFY_SYSTEM = f"""
You classify financial symbols. Use web search to identify sector, country of listing/primary exposure, and instrument class.
Rules:
- Be factual and conservative. If uncertain, set fields to null and lower confidence.
- Output exactly ONE JSON array wrapped by these sentinels:
{S}
[{{"symbol":"...", "sector":"...", "country":"US","asset_class":"equity|etf|crypto|bond|fund|commodity|other",
   "instrument":"single_stock|etf|mutual_fund|crypto|fixed_income|cash|other", "region":"US|CA|EU|EM|ASIA|GLOBAL|OTHER",
   "confidence":0.0-1.0, "citations":[{{"title":"...","url":"https://..."}}]}}]
{E}
- Keep strings short; prefer GICS-like sector names when applicable.
""".strip()

def classify_symbols_pplx(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}
    uniq = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    payload = {"task": "Classify symbols", "symbols": uniq}

    resp = _client.chat.completions.create(
        model=PPLX_MODEL,
        temperature=0.0,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        extra_body={"search_recency_filter": SEARCH_RECENCY, "return_images": False},
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        try:
            raw = extract_json(text, expect="any")  # may be list | dict | str
        except Exception:
            raw = text

        items = _coerce_classification_items(raw)  # always List[Dict[str, Any]]
        out: Dict[str, Dict[str, Any]] = {}
        for item in items:
            # item is narrowed to dict -> Pylance OK
            sym = str(item.get("symbol") or "").upper().strip()
            if not sym:
                continue
            out[sym] = {
                "sector": _none_if_empty(item.get("sector")),
                "country": _norm_country(item.get("country")),
                "asset_class": _none_if_empty(item.get("asset_class")),
                "instrument": _none_if_empty(item.get("instrument")),
                "region": _none_if_empty(item.get("region")),
                "confidence": float(item.get("confidence") or 0.0),
                "citations": _norm_citations(item.get("citations")),
            }
        return out
    except Exception:
        # fail-soft: return empty -> positions stay without sector/country
        return {}

# ---------- AI: portfolio insights & narrative ----------

INSIGHTS_SYSTEM = f"""
You are a portfolio coach. Generate tailored insights from the provided portfolio data ONLY.
Rules:
- Do not invent prices or holdings beyond inputs.
- Insights must be actionable and prioritized.
- Output exactly ONE JSON object wrapped by these sentinels:
{S}
{{"insights":[{{"type":"warning|opportunity|positive","title":"...","description":"...","priority":"high|medium|low","action":"...","related_symbols":["SYM",...]}}]],
 "narrative_markdown":"..."}}
{E}
- Keep each description ≤ 280 chars. Prefer concrete references to weights, concentration, winners/losers, sectors, countries.
- If data is sparse, say what's missing and give best-effort guidance.
""".strip()

def _pnl_for_row(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    qty, buy, cur = r.get("quantity"), r.get("purchase_price"), r.get("current_price")
    if not (isinstance(qty, (int, float)) and isinstance(buy, (int, float)) and isinstance(cur, (int, float))):
        return None, None
    cost = float(qty) * float(buy)
    mv   = float(qty) * float(cur)
    pnl_abs = mv - cost
    pnl_pct = ((mv / cost) - 1.0) * 100.0 if cost > 0 else None
    return pnl_abs, pnl_pct

def _winners_losers(rows: List[Dict[str, Any]], limit: int = 3):
    scored = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if not sym: continue
        _, pnl_pct = _pnl_for_row(r)
        if pnl_pct is None: continue
        scored.append({"symbol": sym, "pnl_pct": round(pnl_pct, 2)})
    winners = sorted([s for s in scored if s["pnl_pct"] >= 0], key=lambda x: x["pnl_pct"], reverse=True)[:limit]
    losers  = sorted([s for s in scored if s["pnl_pct"] <  0], key=lambda x: x["pnl_pct"])[:limit]
    return winners, losers

def generate_portfolio_insights_pplx(
    *, summary: Dict[str, Any], positions: List[Dict[str, Any]], rows: List[Dict[str, Any]],
    portfolio_return_pct: Optional[float], benchmark_return_pct: Optional[float]
) -> Tuple[List[Dict[str, Any]], str]:
    winners, losers = _winners_losers(rows)
    payload = {
        "summary": summary,
        "positions": [
            {"symbol": p["symbol"], "weight": round(float(p["weight"]), 6), "sector": p.get("sector"), "country": p.get("country")}
            for p in positions
        ],
        "performance": {
            "portfolio_return_pct": portfolio_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "winners": winners,
            "losers": losers,
        },
        "notes": "Use only the fields above. No extra facts.",
    }

    resp = _client.chat.completions.create(
        model=PPLX_MODEL,              # prefer sonar-reasoning-pro if you have it
        temperature=0.3,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": INSIGHTS_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        extra_body={"search_recency_filter": SEARCH_RECENCY, "return_images": False},
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        j = extract_json(text)  # {"insights": [...], "narrative_markdown": "..."}
        insights = j.get("insights") or []
        # Normalize insight items and add auto IDs
        normalized: List[Dict[str, Any]] = []
        nid = 1
        for it in insights:
            if not isinstance(it, dict): continue
            _type = it.get("type")
            if _type not in ("warning", "opportunity", "positive"): continue
            normalized.append({
                "id": nid,
                "type": _type,
                "title": it.get("title") or "Insight",
                "description": it.get("description") or "",
                "priority": it.get("priority") if it.get("priority") in ("high","medium","low") else "medium",
                "action": it.get("action") or "View Details",
                "related_symbols": it.get("related_symbols") or [],
            })
            nid += 1
        narrative_md = (j.get("narrative_markdown") or "").strip()
        return normalized, narrative_md
    except Exception:
        # fail-soft fallback if parsing fails
        fallback = [{
            "id": 1, "type": "opportunity", "title": "Review Portfolio",
            "description": "We generated your weights and concentration. Consider rebalancing if a single holding dominates.",
            "priority": "medium", "action": "Open Breakdown", "related_symbols": []
        }]
        return fallback, ""

# ---------- Deterministic internals (weights/HHI) ----------

def _row_value(row: Dict[str, Any]) -> Optional[float]:
    v = row.get("value")
    if isinstance(v, (int, float)) and v >= 0:
        return float(v)
    qty, px = row.get("quantity"), row.get("current_price")
    if isinstance(qty, (int, float)) and isinstance(px, (int, float)):
        return max(0.0, float(qty) * float(px))
    return None

def _totals_by_symbol(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for r in rows:
        sym = str(r.get("symbol") or "").upper().strip()
        if not sym:
            continue
        val = _row_value(r)
        if val is None:
            continue
        totals[sym] += val
    return dict(totals)

def _positions_from_totals(totals: Dict[str, float]) -> List[Position]:
    if not totals:
        return []
    total_val = sum(v for v in totals.values() if v is not None)
    if total_val <= 0:
        syms = sorted(totals.keys())
        n = len(syms)
        if not n:
            return []
        w = 1.0 / n
        return [Position(s, w) for s in syms]

    res = [Position(symbol=s, weight=float(v) / total_val) for s, v in totals.items()]
    s = sum(p.weight for p in res)
    if s > 0:
        for p in res:
            p.weight /= s
    return res

def _summary(positions: List[Position]) -> Dict[str, Any]:
    by_sector: Dict[str, float] = defaultdict(float)
    by_country: Dict[str, float] = defaultdict(float)

    for p in positions:
        sec = (p.sector or "Unknown").strip().title()
        cty = (p.country or "Unknown").strip().upper()
        by_sector[sec] += p.weight
        by_country[cty] += p.weight

    _renorm(by_sector); _renorm(by_country)
    hhi = sum(p.weight ** 2 for p in positions) if positions else 0.0
    top = max(positions, key=lambda p: p.weight) if positions else None
    top_pair = (top.symbol, top.weight) if top else ("", 0.0)

    return {"by_sector": dict(by_sector), "by_country": dict(by_country), "hhi": hhi, "top_position": top_pair}

def _renorm(d: Dict[str, float]) -> None:
    s = sum(d.values())
    if s > 0:
        for k in list(d.keys()):
            d[k] = d[k] / s

def _clean_or(cur: Optional[str], new: Optional[str]) -> Optional[str]:
    val = (new or cur)
    if not val:
        return None
    v = str(val).strip()
    return v if v else None

def _norm_country(v: Any) -> Optional[str]:
    if not v:
        return None
    c = str(v).strip().upper()
    aliases = {"USA": "US", "U.S.": "US", "UNITED STATES": "US", "UK": "GB", "U.K.": "GB"}
    return aliases.get(c, c)

def _safe_analyze(fn: Callable[[str], Dict[str, Any]], symbol: str) -> Dict[str, Any]:
    try:
        return fn(symbol)
    except Exception as e:
        return {"error": "analysis_failed", "detail": str(e)}
    
def _none_if_empty(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def _norm_citations(v: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(v, list):
        for c in v:
            if isinstance(c, dict):
                title = str(c.get("title") or "")
                url = str(c.get("url") or "")
                if url:
                    out.append({"title": title, "url": url})
            elif isinstance(c, str):
                # sometimes models return just URLs
                out.append({"title": "", "url": c})
    return out

def _coerce_classification_items(val: Any) -> List[Dict[str, Any]]:
    """
    Normalize model output to a list[dict]. Accepts:
      - list[dict]            -> as-is
      - dict with 'items'     -> list
      - dict with 'data'      -> list/data
      - single dict           -> [dict]
      - JSON string           -> json.loads then recurse
      - anything else         -> []
    """
    import json
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    if isinstance(val, dict):
        if isinstance(val.get("items"), list):
            return [x for x in val["items"] if isinstance(x, dict)]
        if isinstance(val.get("data"), list):
            return [x for x in val["data"] if isinstance(x, dict)]
        return [val]
    if isinstance(val, str):
        try:
            j = json.loads(val)
            return _coerce_classification_items(j)
        except Exception:
            return []
    return []