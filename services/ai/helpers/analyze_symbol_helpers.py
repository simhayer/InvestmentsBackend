import hashlib
from typing import Any, Dict, List
from utils.common_helpers import safe_float
import asyncio
from services.yahoo_service import get_price_history
# services/ai/helpers/analyze_symbol_io.py
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, cast

from database import get_db
from services.cache.cache_backend import cache_get, cache_set
from services.vector.vector_store_service import VectorStoreService
from utils.common_helpers import timed, fmt_pct
from services.ai.analyze_symbol.types import AgentState

logger = logging.getLogger("analysis_timing")


async def fetch_history_points(symbol: str, period="1y", interval="1d"):
    res = await asyncio.to_thread(get_price_history, symbol, period, interval)
    if isinstance(res, dict) and res.get("status") == "ok":
        return res.get("points") or []
    return []

def compute_market_snapshot(finnhub_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic snapshot based on your Finnhub fundamentals output:
    finnhub_data = {symbol, profile, quote, metrics, earnings, normalized}
    """
    normalized = (finnhub_data or {}).get("normalized") or {}

    mc = safe_float(normalized.get("market_cap"))
    pe = safe_float(normalized.get("pe_ttm"))
    rg = safe_float(normalized.get("revenue_growth_yoy"))
    gm = safe_float(normalized.get("gross_margin"))
    om = safe_float(normalized.get("operating_margin"))
    fcf = safe_float(normalized.get("free_cash_flow"))
    de = safe_float(normalized.get("debt_to_equity"))

    snap: Dict[str, Any] = {
        "market_cap": mc,
        "pe_ttm": pe,
        "revenue_growth_yoy": rg,
        "gross_margin": gm,
        "operating_margin": om,
        "free_cash_flow": fcf,
        "debt_to_equity": de,
    }

    flags: List[str] = []

    if pe is None:
        flags.append("Valuation: PE not available")
    elif pe >= 35:
        flags.append("Valuation: High PE (market likely expects strong execution)")
    elif pe <= 12:
        flags.append("Valuation: Low PE (market likely pricing in risk/low growth)")
    else:
        flags.append("Valuation: Mid-range PE")

    if rg is None:
        flags.append("Growth: Revenue growth not available")
    elif rg >= 15:
        flags.append("Growth: High revenue growth")
    elif rg <= 0:
        flags.append("Growth: Flat/negative revenue growth")
    else:
        flags.append("Growth: Moderate revenue growth")

    if om is None:
        flags.append("Profitability: Operating margin not available")
    elif om >= 0.25:
        flags.append("Profitability: Strong operating margin")
    elif om <= 0.10:
        flags.append("Profitability: Thin operating margin")
    else:
        flags.append("Profitability: Moderate operating margin")

    if fcf is None:
        flags.append("Cash flow: Free cash flow not available")
    elif fcf > 0:
        flags.append("Cash flow: Positive free cash flow")
    elif fcf < 0:
        flags.append("Cash flow: Negative free cash flow")
    else:
        flags.append("Cash flow: Flat free cash flow")

    if de is None:
        flags.append("Leverage: Debt-to-equity not available")
    elif de >= 1.5:
        flags.append("Leverage: Elevated leverage")
    elif de <= 0.5:
        flags.append("Leverage: Conservative leverage")
    else:
        flags.append("Leverage: Moderate leverage")

    snap["flags"] = flags
    return snap

def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def preview(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def ck_fund(symbol: str) -> str:
    return f"ANALYZE:FUND:{normalize_symbol(symbol)}"


def ck_tav(symbol: str) -> str:
    return f"ANALYZE:TAV:{normalize_symbol(symbol)}"


def ck_task(task_id: str) -> str:
    return f"ANALYZE:TASK:{(task_id or '').strip()}"


def ck_report(symbol: str) -> str:
    return f"ANALYZE:REPORT:{normalize_symbol(symbol)}"


@contextmanager
def db_session():
    """
    Wrap get_db() generator safely.
    """
    db_gen = get_db()
    try:
        db = next(db_gen)
        yield db
    finally:
        # exhaust/close generator
        try:
            next(db_gen, None)
        except Exception:
            pass


def format_sec_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No specific SEC section context found."
    lines: List[str] = []
    for c in chunks:
        content = c.get("content")
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines) if lines else "No specific SEC section context found."


async def get_fundamentals_with_cache(
    *,
    symbol: str,
    state: AgentState,
    ttl_seconds: int,
    fetch_fundamentals_cached,  # injected to keep file decoupled
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Cache contract preserved:
      cache value: {"data": finnhub_data, "gaps": finnhub_gaps}
    """
    key = ck_fund(symbol)
    cached = cache_get(key)

    if isinstance(cached, dict) and "data" in cached and "gaps" in cached:
        finnhub_data = cached.get("data") or {}
        finnhub_gaps = cached.get("gaps") or []
        return finnhub_data, finnhub_gaps

    with timed("finnhub_fundamentals", logger, state=state):
        finres = await fetch_fundamentals_cached(symbol, timeout_s=5.0)

    finnhub_data = finres.data
    finnhub_gaps = finres.gaps
    cache_set(key, {"data": finnhub_data, "gaps": finnhub_gaps}, ttl_seconds=ttl_seconds)
    return finnhub_data, finnhub_gaps


async def get_news_with_optional_tavily_fallback(
    *,
    symbol: str,
    state: AgentState,
    ttl_tavily_seconds: int,
    get_company_news_cached,     # injected
    tavily_search=None,          # injected
    compact_tavily=None,         # injected
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Preserves behavior:
      - tries Finnhub news first (days_back=10, limit=10)
      - if <3 items, tries Tavily with the SAME query string
      - caches Tavily raw_str under ANALYZE:TAV:<SYMBOL>
    """
    news_items: List[Dict[str, Any]] = []
    news_compact = ""
    used_tavily = False

    try:
        with timed("finnhub_news", logger, state=state):
            news_payload = await get_company_news_cached(symbol, days_back=10, limit=10)
        news_items = news_payload.get("items") or []
        news_compact = news_payload.get("compact") or ""
    except Exception:
        news_items = []
        news_compact = ""

    raw_str = news_compact

    if (len(news_items) < 3) and tavily_search is not None and compact_tavily is not None:
        tav_key = ck_tav(symbol)
        cached_t = cache_get(tav_key)

        if isinstance(cached_t, dict) and isinstance(cached_t.get("raw"), str):
            raw_str = cached_t["raw"]
            used_tavily = True
        else:
            try:
                with timed("tavily_search", logger, state=state):
                    results = await tavily_search(
                        query=f"latest news, upcoming catalysts, bull and bear case for {symbol}",
                        max_results=8,
                        search_depth="advanced",
                    )
                raw_str = compact_tavily(results)
                cache_set(tav_key, {"raw": raw_str}, ttl_seconds=ttl_tavily_seconds)
                used_tavily = True
            except Exception:
                raw_str = news_compact

    return news_items, raw_str, used_tavily


def get_sec_routed_context(
    *,
    symbol: str,
    state: AgentState,
    task_id: str,
    vs: Optional[VectorStoreService] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Preserves behavior:
      - same section names: Item 1, Item 1A, Item 7
      - same limits: 5, 8, 5
      - builds combined sec_context up to 12000 chars
    """
    sec_business: List[Dict[str, Any]] = []
    sec_risks: List[Dict[str, Any]] = []
    sec_mda: List[Dict[str, Any]] = []
    sec_context = ""
    sec_debug = {"count": 0, "chunks": []}

    try:
        with db_session() as db:
            vs = vs or VectorStoreService()

            with timed("sec_vector_routing", logger, state=state):
                sec_business = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 1", limit=5)
                sec_risks = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 1A", limit=8)
                sec_mda = vs.get_context_for_analysis(db, symbol, "...", section_name="Item 7", limit=5)

            logger.debug(
                "[%s] SEC routing for %s: sec_business=%s sec_risks=%s sec_mda=%s",
                task_id, symbol, len(sec_business), len(sec_risks), len(sec_mda)
            )

            all_chunks = sec_business + sec_risks + sec_mda
            sec_debug["count"] = len(all_chunks)
            sec_debug["chunks"] = [
                {"score": c.get("score"), "section": (c.get("metadata") or {}).get("section_name")}
                for c in all_chunks[:10]
            ]

            sec_context = "\n".join([f"- {c.get('content')}" for c in all_chunks])[:12000]

    except Exception as e:
        logger.warning(f"[{task_id}] SEC routing failed for {symbol}: {e}")
        sec_context = ""

    return sec_context, sec_business, sec_risks, sec_mda, sec_debug


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)




def shorten_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_chars else (s[: max_chars - 1] + "…")


def trim_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    # keep only what you actually use in your report
    if not profile:
        return {}
    keep = ["name", "ticker", "exchange", "country", "currency", "sector", "industry", "marketCapitalization"]
    out = {k: profile.get(k) for k in keep if profile.get(k) not in (None, "", "NA")}
    return out


def trim_quote(quote: Dict[str, Any]) -> Dict[str, Any]:
    if not quote:
        return {}
    keep = ["c", "pc", "d", "dp", "h", "l", "o", "t"]
    out = {k: quote.get(k) for k in keep if quote.get(k) not in (None, "", "NA")}
    return out


def trim_market_snapshot(ms: Dict[str, Any]) -> Dict[str, Any]:
    if not ms:
        return {}
    # keep this small; avoid verbose narrative fields
    keep = ["as_of", "price", "prev_close", "day_change_pct", "market_cap", "pe_ttm", "beta", "dividend_yield"]
    out = {k: ms.get(k) for k in keep if ms.get(k) not in (None, "", "NA")}
    return out


def trim_earnings_calendar(cal: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    if not cal:
        return []
    out: List[Dict[str, Any]] = []
    for e in cal[:max_items]:
        out.append({
            "date": e.get("date"),
            "eps_estimate": e.get("epsEstimate"),
            "revenue_estimate": e.get("revenueEstimate"),
            "quarter": e.get("quarter"),
            "year": e.get("year"),
        })
    return out


def trim_sec_chunks(chunks: List[Any], max_chunks: int = 2, max_chars_each: int = 650) -> List[Dict[str, str]]:
    """
    Your SEC chunks might be strings or dict-like objects depending on VectorStoreService.
    We convert to small dicts with 'text' only.
    """
    if not chunks:
        return []
    trimmed: List[Dict[str, str]] = []
    for c in chunks[:max_chunks]:
        if isinstance(c, dict):
            txt = c.get("text") or c.get("chunk") or c.get("content") or ""
        else:
            txt = str(c)
        txt = shorten_text(txt, max_chars_each)
        if txt:
            trimmed.append({"text": txt})
    return trimmed


def trim_news_items(news_items: List[Dict[str, Any]], max_items: int = 5, max_chars_each: int = 220) -> List[Dict[str, str]]:
    if not news_items:
        return []
    out: List[Dict[str, str]] = []
    for n in news_items[:max_items]:
        title = shorten_text(str(n.get("headline") or n.get("title") or ""), 120)
        date = str(n.get("datetime") or n.get("date") or "")
        summary = shorten_text(str(n.get("summary") or n.get("snippet") or ""), max_chars_each)
        item = {
            "date": date,
            "title": title,
            "summary": summary,
        }
        # keep it clean: drop empty fields
        out.append({k: v for k, v in item.items() if v})
    return out


def stable_hash(obj: Any) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(obj)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d.get(k)
    return default

def build_technicals_text_from_pack(symbol: str, tech_pack: Dict[str, Any]) -> str:
    """
    Builds a short "Market Performance" paragraph from your computed tech_pack.
    This assumes tech_pack contains common fields; it degrades gracefully if missing.

    Expected-ish keys (adjust if your pack differs):
      - returns: {"1D": x, "1M": x, "1Y": x} or similar
      - trend: {"above_ma50": bool, "above_ma200": bool} or ma fields
      - relative: {"vs_spy_1y": x} or similar
    """
    returns = _get(tech_pack, "returns", default={}) or {}
    r1d = returns.get("1D") or returns.get("1d")
    r1m = returns.get("1M") or returns.get("1m")
    r1y = returns.get("1Y") or returns.get("1y")

    # Trend flags (try a few common shapes)
    trend = _get(tech_pack, "trend", default={}) or {}
    above_50 = trend.get("above_ma50")
    above_200 = trend.get("above_ma200")

    # If your pack stores MAs as numbers, infer
    price = _get(tech_pack, "last_price", "price", default=None)
    ma50 = _get(tech_pack, "ma50", "sma50", default=None)
    ma200 = _get(tech_pack, "ma200", "sma200", default=None)
    if above_50 is None and price is not None and ma50 is not None:
        above_50 = price >= ma50
    if above_200 is None and price is not None and ma200 is not None:
        above_200 = price >= ma200

    rel = _get(tech_pack, "relative", default={}) or {}
    vs_spy_1y = rel.get("vs_spy_1y") or rel.get("vsSPY_1Y") or rel.get("vs_spy")

    pieces: List[str] = []
    pieces.append(
        f"{symbol} returns: 1D {fmt_pct(r1d)}, 1M {fmt_pct(r1m)}, 1Y {fmt_pct(r1y)}."
    )

    if above_50 is not None or above_200 is not None:
        t = []
        if above_50 is not None:
            t.append("above the 50-day MA" if above_50 else "below the 50-day MA")
        if above_200 is not None:
            t.append("above the 200-day MA" if above_200 else "below the 200-day MA")
        pieces.append("Trend: " + ", ".join(t) + ".")

    if vs_spy_1y is not None:
        pieces.append(f"Relative vs SPY (1Y): {fmt_pct(vs_spy_1y)}.")

    return " ".join(pieces).strip()


REQ_CATALYST_FIELDS = [
    "name", "window", "trigger", "mechanism", "likely_market_reaction",
    "impact_channels", "probability", "magnitude", "priced_in", "key_watch_items"
]

def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""

def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def _is_str_or_none(x):
    return x is None or isinstance(x, str)

def _req_nonempty_str(d, key, issues, msg):
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        issues.append(msg)

def validate_report(report: Dict[str, Any], *, next_earnings_date: str, finnhub_gaps: List[str]) -> List[str]:
    issues: List[str] = []

    if not isinstance(report, dict):
        return ["Report is not a dict."]

    # --- recommendation / confidence ---
    rec = report.get("recommendation")
    if rec not in ("Buy", "Hold", "Sell"):
        issues.append("recommendation must be Buy/Hold/Sell.")

    conf = report.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
        issues.append("confidence must be a number between 0 and 1.")
        conf = None
    else:
        if finnhub_gaps and conf > 0.55:
            issues.append("confidence too high given FINNHUB gaps (cap 0.55).")

    # --- key_insights (now objects) ---
    key_insights = _as_list(report.get("key_insights"))
    if len(key_insights) < 3:
        issues.append("key_insights should have at least 3 items.")
    else:
        for i, ki in enumerate(key_insights, 1):
            if not isinstance(ki, dict):
                issues.append(f"key_insight #{i} must be an object.")
                continue
            if not isinstance(ki.get("insight"), str) or not ki["insight"].strip():
                issues.append(f"key_insight #{i} missing non-empty insight.")
            if not _is_str_or_none(ki.get("evidence")):
                issues.append(f"key_insight #{i} evidence must be string or null.")
            if not _is_str_or_none(ki.get("implication")):
                issues.append(f"key_insight #{i} implication must be string or null.")

    # --- risks ---
    if len(_as_list(report.get("stock_overflow_risks"))) < 2:
        issues.append("stock_overflow_risks should have at least 2 items.")

    # --- scenarios ---
    scenarios = _as_list(report.get("scenarios"))
    names: List[str] = [cast(str, s.get("name")) for s in scenarios if isinstance(s, dict) and isinstance(s.get("name"), str)]
    if len(scenarios) != 3 or sorted(names) != ["Base", "Bear", "Bull"]:
        issues.append("scenarios must be exactly 3 with names Base/Bull/Bear.")

    # --- catalysts ---
    catalysts = _as_list(report.get("upcoming_catalysts"))
    if len(catalysts) != 3:
        issues.append("upcoming_catalysts must be exactly 3 items.")
    else:
        for i, c in enumerate(catalysts, start=1):
            if not isinstance(c, dict):
                issues.append(f"catalyst #{i} is not an object.")
                continue
            missing = [f for f in REQ_CATALYST_FIELDS if f not in c]
            if missing:
                issues.append(f"catalyst #{i} missing fields: {', '.join(missing)}")

            window = c.get("window")
            if not _is_nonempty_str(window):
                issues.append(f"catalyst #{i} window is empty.")
            else:
                name = (c.get("name") or "").lower()
                trig = (c.get("trigger") or "").lower()
                if ("earnings" in name) or ("earnings" in trig):
                    if next_earnings_date != "Unknown" and window != next_earnings_date:
                        issues.append(f"earnings catalyst window must equal {next_earnings_date} exactly.")

            p = c.get("probability")
            if not isinstance(p, (int, float)) or p < 0.0 or p > 1.0:
                issues.append(f"catalyst #{i} probability must be 0..1.")

    # --- market_edge rules tied to confidence ---
    if isinstance(conf, (int, float)) and conf >= 0.6:
        me = report.get("market_edge")
        if not isinstance(me, dict):
            issues.append("market_edge must be present when confidence >= 0.6.")
        else:
            _req_nonempty_str(me, "consensus_view", issues, "market_edge.consensus_view must be non-empty.")
            _req_nonempty_str(me, "variant_view", issues, "market_edge.variant_view must be non-empty.")
            _req_nonempty_str(me, "why_it_matters", issues, "market_edge.why_it_matters must be non-empty.")

    # --- is_priced_in logic ---
    is_priced_in = report.get("is_priced_in")
    if not isinstance(is_priced_in, bool):
        issues.append("is_priced_in must be boolean.")
    else:
        pa = report.get("pricing_assessment")
        if is_priced_in is False:
            if not isinstance(pa, dict):
                issues.append("pricing_assessment must be present when is_priced_in = false.")
            else:
                # choose keys you want to standardize:
                for k in ("market_expectation", "variant_outcome", "valuation_sensitivity"):
                    if not _is_nonempty_str(pa.get(k)):
                        issues.append(f"pricing_assessment.{k} must be non-empty when is_priced_in = false.")

    # --- thesis_points cap (keep yours) ---
    thesis_points = _as_list(report.get("thesis_points"))
    if thesis_points and len(thesis_points) > 5:
        issues.append("thesis_points should be <= 5 for speed/brevity.")

    return issues