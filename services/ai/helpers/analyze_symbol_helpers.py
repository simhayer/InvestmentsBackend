import hashlib
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple, cast
from contextlib import contextmanager
from functools import lru_cache

from database import get_db
from services.cache.cache_backend import cache_get, cache_set
from services.vector.vector_store_service import VectorStoreService
from services.yahoo_service import get_price_history
from utils.common_helpers import timed, fmt_pct, safe_float
from services.ai.analyze_symbol.types import AgentState

logger = logging.getLogger("analysis_timing")

# ============================================================================
# CONSTANTS
# ============================================================================

# Cache key prefixes
CACHE_PREFIX_FUND = "ANALYZE:FUND:"
CACHE_PREFIX_TAV = "ANALYZE:TAV:"
CACHE_PREFIX_TASK = "ANALYZE:TASK:"
CACHE_PREFIX_REPORT = "ANALYZE:REPORT:"
CACHE_PREFIX_TECH = "ANALYZE:TECH:"

# Data limits
MAX_SEC_CONTEXT_CHARS = 12000
MAX_SEC_BUSINESS_CHUNKS = 5
MAX_SEC_RISK_CHUNKS = 8
MAX_SEC_MDA_CHUNKS = 5

# Validation thresholds
MIN_KEY_INSIGHTS = 3
MIN_KEY_RISKS = 3
REQUIRED_SCENARIOS = 3
REQUIRED_CATALYSTS = 3
MAX_THESIS_POINTS = 5

# Quality thresholds
MIN_SEC_CONTEXT_LENGTH = 80
MIN_NEWS_COUNT_THRESHOLD = 3
MIN_METRICS_REQUIRED = 4

def normalize_symbol(symbol: str) -> str:
    """Normalize ticker symbol to uppercase."""
    return (symbol or "").strip().upper()


def ck_fund(symbol: str) -> str:
    """Cache key for fundamentals data."""
    return f"{CACHE_PREFIX_FUND}{normalize_symbol(symbol)}"


def ck_tav(symbol: str) -> str:
    """Cache key for Tavily search results."""
    return f"{CACHE_PREFIX_TAV}{normalize_symbol(symbol)}"


def ck_task(task_id: str) -> str:
    """Cache key for analysis task status."""
    return f"{CACHE_PREFIX_TASK}{(task_id or '').strip()}"


def ck_report(symbol: str) -> str:
    """Cache key for analysis report."""
    return f"{CACHE_PREFIX_REPORT}{normalize_symbol(symbol)}"


def ck_tech(symbol: str) -> str:
    """Cache key for technical analysis."""
    return f"{CACHE_PREFIX_TECH}{normalize_symbol(symbol)}"


def preview(s: str, n: int = 220) -> str:
    """Generate preview of text with ellipsis if truncated."""
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def shorten_text(s: str, max_chars: int) -> str:
    """Shorten text to max_chars with ellipsis."""
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= max_chars else (s[:max_chars - 1] + "…")


def stable_hash(obj: Any) -> str:
    """Generate stable hash for any object (for cache invalidation)."""
    try:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(obj)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def json_dumps(obj: Any) -> str:
    """JSON serialization with unicode support."""
    return json.dumps(obj, ensure_ascii=False)


@contextmanager
def db_session():
    """
    Safely wrap get_db() generator.
    Ensures proper cleanup even on exceptions.
    """
    db_gen = get_db()
    try:
        db = next(db_gen)
        yield db
    finally:
        try:
            next(db_gen, None)
        except (StopIteration, Exception):
            pass


def _get(d: Dict[str, Any], *keys, default=None):
    """Get first matching key from dict."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d.get(k)
    return default

async def fetch_history_points(
    symbol: str, 
    period: str = "1y", 
    interval: str = "1d"
) -> List[Dict[str, Any]]:
    try:
        res = await asyncio.to_thread(get_price_history, symbol, period, interval)
        if isinstance(res, dict) and res.get("status") == "ok":
            return res.get("points") or []
    except Exception as e:
        logger.error(f"Failed to fetch history for {symbol}: {e}")
    return []


async def get_fundamentals_with_cache(
    *,
    symbol: str,
    state: AgentState,
    ttl_seconds: int,
    fetch_fundamentals_cached,
) -> Tuple[Dict[str, Any], List[str]]:
    key = ck_fund(symbol)
    cached = cache_get(key)

    # Check cache validity
    if isinstance(cached, dict) and "data" in cached and "gaps" in cached:
        finnhub_data = cached.get("data") or {}
        finnhub_gaps = cached.get("gaps") or []
        logger.debug(f"Cache hit for fundamentals: {symbol}")
        return finnhub_data, finnhub_gaps

    # Fetch fresh data
    with timed("finnhub_fundamentals", logger, state=state):
        try:
            finres = await fetch_fundamentals_cached(symbol, timeout_s=5.0)
            finnhub_data = finres.data
            finnhub_gaps = finres.gaps
        except Exception as e:
            logger.error(f"Fundamentals fetch failed for {symbol}: {e}")
            return {}, ["fundamentals_fetch_failed"]

    # Cache result
    cache_set(
        key, 
        {"data": finnhub_data, "gaps": finnhub_gaps}, 
        ttl_seconds=ttl_seconds
    )
    
    return finnhub_data, finnhub_gaps


async def get_news_with_optional_tavily_fallback(
    *,
    symbol: str,
    state: AgentState,
    ttl_tavily_seconds: int,
    get_company_news_cached,
    tavily_search=None,
    compact_tavily=None,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    news_items: List[Dict[str, Any]] = []
    news_compact = ""
    used_tavily = False

    # Try Finnhub first
    try:
        with timed("finnhub_news", logger, state=state):
            news_payload = await get_company_news_cached(
                symbol, 
                days_back=10, 
                limit=15
            )
        news_items = news_payload.get("items") or []
        news_compact = news_payload.get("compact") or ""
    except Exception as e:
        logger.warning(f"Finnhub news failed for {symbol}: {e}")
        news_items = []
        news_compact = ""

    raw_str = news_compact

    # Fallback to Tavily if insufficient news
    if len(news_items) < MIN_NEWS_COUNT_THRESHOLD:
        if tavily_search is None or compact_tavily is None:
            logger.debug(f"Tavily not available for {symbol}")
            return news_items, raw_str, used_tavily

        tav_key = ck_tav(symbol)
        cached_t = cache_get(tav_key)

        # Check Tavily cache
        if isinstance(cached_t, dict) and isinstance(cached_t.get("raw"), str):
            raw_str = cached_t["raw"]
            used_tavily = True
            logger.debug(f"Tavily cache hit for {symbol}")
        else:
            # Fetch from Tavily
            try:
                with timed("tavily_search", logger, state=state):
                    results = await tavily_search(
                        query=f"latest news, upcoming catalysts, bull and bear case for {symbol}",
                        max_results=8,
                        search_depth="advanced",
                    )
                raw_str = compact_tavily(results)
                cache_set(
                    tav_key, 
                    {"raw": raw_str}, 
                    ttl_seconds=ttl_tavily_seconds
                )
                used_tavily = True
                logger.info(f"Tavily fallback used for {symbol}")
            except Exception as e:
                logger.warning(f"Tavily search failed for {symbol}: {e}")
                raw_str = news_compact

    return news_items, raw_str, used_tavily


def get_sec_routed_context(
    *,
    symbol: str,
    state: AgentState,
    task_id: str,
    vs: Optional[VectorStoreService] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sec_business: List[Dict[str, Any]] = []
    sec_risks: List[Dict[str, Any]] = []
    sec_mda: List[Dict[str, Any]] = []
    sec_context = ""
    sec_debug = {"count": 0, "chunks": [], "sections_found": {}}

    try:
        with db_session() as db:
            vs = vs or VectorStoreService()

            # Lazy-load SEC vectors if missing for the symbol
            try:
                if not vs.has_symbol(db, symbol):
                    logger.info("[%s] SEC vectors missing for %s. Populating now.", task_id, symbol)
                    from services.filings.filing_service import FilingService
                    FilingService().process_company_filings_task(symbol)
                    try:
                        db.expire_all()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("[%s] SEC lazy load failed for %s: %s", task_id, symbol, e)

            with timed("sec_vector_routing", logger, state=state):
                # Parallel fetching of different sections
                sec_business = vs.get_context_for_analysis(
                    db, symbol, "...", 
                    section_name="Item 1", 
                    limit=MAX_SEC_BUSINESS_CHUNKS
                )
                sec_risks = vs.get_context_for_analysis(
                    db, symbol, "...", 
                    section_name="Item 1A", 
                    limit=MAX_SEC_RISK_CHUNKS
                )
                sec_mda = vs.get_context_for_analysis(
                    db, symbol, "...", 
                    section_name="Item 7", 
                    limit=MAX_SEC_MDA_CHUNKS
                )

            # Build debug info
            sec_debug["sections_found"] = {
                "Item 1": len(sec_business),
                "Item 1A": len(sec_risks),
                "Item 7": len(sec_mda),
            }

            logger.debug(
                "[%s] SEC routing for %s: business=%d risks=%d mda=%d",
                task_id, symbol, 
                len(sec_business), len(sec_risks), len(sec_mda)
            )

            # Combine all chunks
            all_chunks = sec_business + sec_risks + sec_mda
            sec_debug["count"] = len(all_chunks)
            sec_debug["chunks"] = [
                {
                    "score": c.get("score"),
                    "section": (c.get("metadata") or {}).get("section_name")
                }
                for c in all_chunks[:10]
            ]

            # Build context text (limited to MAX_SEC_CONTEXT_CHARS)
            sec_context = "\n".join([
                f"- {c.get('content')}" 
                for c in all_chunks
            ])[:MAX_SEC_CONTEXT_CHARS]

    except Exception as e:
        logger.warning(f"[{task_id}] SEC routing failed for {symbol}: {e}")
        sec_debug["error"] = str(e)
        sec_context = ""

    return sec_context, sec_business, sec_risks, sec_mda, sec_debug

def compute_market_snapshot(finnhub_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = (finnhub_data or {}).get("normalized") or {}

    # Extract metrics
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

    # Generate interpretive flags
    flags: List[str] = []

    # Valuation analysis
    if pe is None:
        flags.append("Valuation: P/E not available")
    elif pe >= 35:
        flags.append("Valuation: High P/E (market expects strong execution)")
    elif pe <= 12:
        flags.append("Valuation: Low P/E (market pricing in risk/low growth)")
    else:
        flags.append("Valuation: Mid-range P/E")

    # Growth analysis
    if rg is None:
        flags.append("Growth: Revenue growth not available")
    elif rg >= 15:
        flags.append("Growth: High revenue growth")
    elif rg <= 0:
        flags.append("Growth: Flat/negative revenue growth")
    else:
        flags.append("Growth: Moderate revenue growth")

    # Profitability analysis
    if om is None:
        flags.append("Profitability: Operating margin not available")
    elif om >= 0.25:
        flags.append("Profitability: Strong operating margin")
    elif om <= 0.10:
        flags.append("Profitability: Thin operating margin")
    else:
        flags.append("Profitability: Moderate operating margin")

    # Cash flow analysis
    if fcf is None:
        flags.append("Cash flow: Free cash flow not available")
    elif fcf > 0:
        flags.append("Cash flow: Positive free cash flow")
    elif fcf < 0:
        flags.append("Cash flow: Negative free cash flow")
    else:
        flags.append("Cash flow: Flat free cash flow")

    # Leverage analysis
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


# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

def compute_data_quality(
    finnhub_data: Dict[str, Any],
    finnhub_gaps: List[str],
    sec_context: str,
    news_count: int
) -> List[str]:
    notes: List[str] = []

    # Check fundamentals gaps
    if finnhub_gaps:
        gap_preview = ", ".join(finnhub_gaps[:10])
        if len(finnhub_gaps) > 10:
            gap_preview += "..."
        notes.append(f"Fundamentals gaps present: {gap_preview}")

    # Check SEC context
    if not sec_context or len(sec_context.strip()) < MIN_SEC_CONTEXT_LENGTH:
        notes.append("SEC filing context is limited or unavailable.")

    # Check news coverage
    if news_count < MIN_NEWS_COUNT_THRESHOLD:
        notes.append(
            "News coverage is light; catalysts may be incomplete or less reliable."
        )

    # Check key metrics availability
    normalized = (finnhub_data or {}).get("normalized") or {}
    common_keys = [
        "market_cap",
        "pe_ttm",
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "debt_to_equity",
    ]
    missing_common = [
        k for k in common_keys 
        if normalized.get(k) in (None, "", "NA")
    ]
    
    if len(missing_common) >= MIN_METRICS_REQUIRED:
        notes.append(
            "Several key normalized metrics are missing; "
            "treat valuation/profitability conclusions cautiously."
        )

    return notes

@lru_cache(maxsize=128)
def _cached_format_pct(value: Optional[float]) -> str:
    """Cached percentage formatting to reduce repeated work."""
    return fmt_pct(value)


def build_technicals_text_from_pack(
    symbol: str, 
    tech_pack: Dict[str, Any]
) -> str:
    returns = _get(tech_pack, "returns", default={}) or {}
    r1d = returns.get("1D") or returns.get("1d")
    r1m = returns.get("1M") or returns.get("1m")
    r1y = returns.get("1Y") or returns.get("1y")

    # Trend indicators
    trend = _get(tech_pack, "trend", default={}) or {}
    above_50 = trend.get("above_ma50")
    above_200 = trend.get("above_ma200")

    # Infer MA positions if not explicitly provided
    price = _get(tech_pack, "last_price", "price", default=None)
    ma50 = _get(tech_pack, "ma50", "sma50", default=None)
    ma200 = _get(tech_pack, "ma200", "sma200", default=None)
    
    if above_50 is None and price is not None and ma50 is not None:
        above_50 = price >= ma50
    if above_200 is None and price is not None and ma200 is not None:
        above_200 = price >= ma200

    # Relative performance
    rel = _get(tech_pack, "relative", default={}) or {}
    vs_spy_1y = (
        rel.get("vs_spy_1y") or 
        rel.get("vsSPY_1Y") or 
        rel.get("vs_spy")
    )

    # Build summary
    pieces: List[str] = []
    
    pieces.append(
        f"{symbol} returns: "
        f"1D {_cached_format_pct(r1d)}, "
        f"1M {_cached_format_pct(r1m)}, "
        f"1Y {_cached_format_pct(r1y)}."
    )

    # Trend analysis
    if above_50 is not None or above_200 is not None:
        trend_parts = []
        if above_50 is not None:
            trend_parts.append(
                "above 50-day MA" if above_50 else "below 50-day MA"
            )
        if above_200 is not None:
            trend_parts.append(
                "above 200-day MA" if above_200 else "below 200-day MA"
            )
        pieces.append("Trend: " + ", ".join(trend_parts) + ".")

    # Relative performance
    if vs_spy_1y is not None:
        pieces.append(f"Relative vs SPY (1Y): {_cached_format_pct(vs_spy_1y)}.")

    return " ".join(pieces).strip()


# ============================================================================
# DATA TRIMMING (for token reduction)
# ============================================================================

def trim_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only essential profile fields."""
    if not profile:
        return {}
    
    essential_fields = [
        "name", "ticker", "exchange", "country", 
        "currency", "sector", "industry", "marketCapitalization"
    ]
    
    return {
        k: profile.get(k) 
        for k in essential_fields 
        if profile.get(k) not in (None, "", "NA")
    }


def trim_quote(quote: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only essential quote fields."""
    if not quote:
        return {}
    
    essential_fields = ["c", "pc", "d", "dp", "h", "l", "o", "t"]
    
    return {
        k: quote.get(k) 
        for k in essential_fields 
        if quote.get(k) not in (None, "", "NA")
    }


def trim_market_snapshot(ms: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only essential snapshot fields."""
    if not ms:
        return {}
    
    essential_fields = [
        "as_of", "price", "prev_close", "day_change_pct",
        "market_cap", "pe_ttm", "beta", "dividend_yield"
    ]
    
    return {
        k: ms.get(k) 
        for k in essential_fields 
        if ms.get(k) not in (None, "", "NA")
    }


def trim_earnings_calendar(
    cal: List[Dict[str, Any]], 
    max_items: int = 3
) -> List[Dict[str, Any]]:
    """Trim earnings calendar to essentials."""
    if not cal:
        return []
    
    trimmed: List[Dict[str, Any]] = []
    for e in cal[:max_items]:
        trimmed.append({
            "date": e.get("date"),
            "eps_estimate": e.get("epsEstimate"),
            "revenue_estimate": e.get("revenueEstimate"),
            "quarter": e.get("quarter"),
            "year": e.get("year"),
        })
    
    return trimmed


def trim_sec_chunks(
    chunks: List[Any],
    max_chunks: int = 2,
    max_chars_each: int = 650
) -> List[Dict[str, str]]:
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


def trim_news_items(
    news_items: List[Dict[str, Any]],
    max_items: int = 5,
    max_chars_each: int = 220
) -> List[Dict[str, str]]:
    """Trim news items for LLM consumption."""
    if not news_items:
        return []
    
    trimmed: List[Dict[str, str]] = []
    for n in news_items[:max_items]:
        title = shorten_text(
            str(n.get("headline") or n.get("title") or ""), 
            120
        )
        date = str(n.get("datetime") or n.get("date") or "")
        summary = shorten_text(
            str(n.get("summary") or n.get("snippet") or ""),
            max_chars_each
        )
        
        item = {
            "date": date,
            "title": title,
            "summary": summary,
        }
        
        # Remove empty fields
        trimmed.append({k: v for k, v in item.items() if v})
    
    return trimmed

# Required catalyst fields
REQ_CATALYST_FIELDS = [
    "name", "window", "trigger", "mechanism", "likely_market_reaction",
    "impact_channels", "probability", "magnitude", "priced_in", "key_watch_items"
]


def _is_nonempty_str(x: Any) -> bool:
    """Check if value is a non-empty string."""
    return isinstance(x, str) and x.strip() != ""


def _as_list(x: Any) -> List[Any]:
    """Safely convert to list."""
    return x if isinstance(x, list) else []


def _is_str_or_none(x: Any) -> bool:
    """Check if value is string or None."""
    return x is None or isinstance(x, str)


def _req_nonempty_str(
    d: Dict[str, Any], 
    key: str, 
    issues: List[str], 
    msg: str
) -> None:
    """Validate required non-empty string field."""
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        issues.append(msg)


def validate_report(
    report: Dict[str, Any],
    *,
    next_earnings_date: str,
    finnhub_gaps: List[str]
) -> List[str]:
    issues: List[str] = []

    if not isinstance(report, dict):
        return ["Report is not a dict."]

    # Validate recommendation
    rec = report.get("recommendation")
    if rec not in ("Buy", "Hold", "Sell"):
        issues.append("recommendation must be Buy/Hold/Sell.")

    # Validate confidence
    conf = report.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
        issues.append("confidence must be a number between 0 and 1.")
        conf = None
    else:
        # Confidence should be lower if data has gaps
        if finnhub_gaps and conf > 0.55:
            issues.append(
                "confidence too high given FINNHUB gaps (should be ≤ 0.55)."
            )

    # Validate key insights (now objects with structured fields)
    key_insights = _as_list(report.get("key_insights"))
    if len(key_insights) < MIN_KEY_INSIGHTS:
        issues.append(f"key_insights should have at least {MIN_KEY_INSIGHTS} items.")
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

    # Validate risks
    if len(_as_list(report.get("key_risks"))) < MIN_KEY_RISKS:
        issues.append(f"key_risks should have at least {MIN_KEY_RISKS} items.")

    # Validate scenarios
    scenarios = _as_list(report.get("scenarios"))
    scenario_names = [
        cast(str, s.get("name")) 
        for s in scenarios 
        if isinstance(s, dict) and isinstance(s.get("name"), str)
    ]
    
    if len(scenarios) != REQUIRED_SCENARIOS or sorted(scenario_names) != ["Base", "Bear", "Bull"]:
        issues.append(
            f"scenarios must be exactly {REQUIRED_SCENARIOS} "
            "with names Base/Bull/Bear."
        )

    # Validate catalysts
    catalysts = _as_list(report.get("upcoming_catalysts"))
    if len(catalysts) != REQUIRED_CATALYSTS:
        issues.append(f"upcoming_catalysts must be exactly {REQUIRED_CATALYSTS} items.")
    else:
        for i, c in enumerate(catalysts, start=1):
            if not isinstance(c, dict):
                issues.append(f"catalyst #{i} is not an object.")
                continue
            
            # Check required fields
            missing = [f for f in REQ_CATALYST_FIELDS if f not in c]
            if missing:
                issues.append(
                    f"catalyst #{i} missing fields: {', '.join(missing)}"
                )

            # Validate window
            window = c.get("window")
            if not _is_nonempty_str(window):
                issues.append(f"catalyst #{i} window is empty.")
            else:
                # Earnings catalyst must match next earnings date
                name = (c.get("name") or "").lower()
                trigger = (c.get("trigger") or "").lower()
                
                if ("earnings" in name) or ("earnings" in trigger):
                    if next_earnings_date != "Unknown" and window != next_earnings_date:
                        issues.append(
                            f"earnings catalyst window must equal "
                            f"{next_earnings_date} exactly."
                        )

            # Validate probability
            prob = c.get("probability")
            if not isinstance(prob, (int, float)) or prob < 0.0 or prob > 1.0:
                issues.append(f"catalyst #{i} probability must be 0..1.")

    # Validate market edge (required for high confidence)
    if isinstance(conf, (int, float)) and conf >= 0.6:
        me = report.get("market_edge")
        if not isinstance(me, dict):
            issues.append("market_edge must be present when confidence >= 0.6.")
        else:
            _req_nonempty_str(
                me, "consensus_view", issues,
                "market_edge.consensus_view must be non-empty."
            )
            _req_nonempty_str(
                me, "variant_view", issues,
                "market_edge.variant_view must be non-empty."
            )
            _req_nonempty_str(
                me, "why_it_matters", issues,
                "market_edge.why_it_matters must be non-empty."
            )

    # Validate pricing assessment
    is_priced_in = report.get("is_priced_in")
    if not isinstance(is_priced_in, bool):
        issues.append("is_priced_in must be boolean.")
    else:
        if is_priced_in is False:
            pa = report.get("pricing_assessment")
            if not isinstance(pa, dict):
                issues.append(
                    "pricing_assessment must be present when is_priced_in = false."
                )
            else:
                required_fields = [
                    "market_expectation",
                    "variant_outcome",
                    "valuation_sensitivity"
                ]
                for field in required_fields:
                    if not _is_nonempty_str(pa.get(field)):
                        issues.append(
                            f"pricing_assessment.{field} must be non-empty "
                            "when is_priced_in = false."
                        )

    # Validate thesis points count
    thesis_points = _as_list(report.get("thesis_points"))
    if thesis_points and len(thesis_points) > MAX_THESIS_POINTS:
        issues.append(
            f"thesis_points should be ≤ {MAX_THESIS_POINTS} for speed/brevity."
        )

    return issues


def format_sec_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format SEC chunks for display."""
    if not chunks:
        return "No specific SEC section context found."
    
    lines: List[str] = []
    for c in chunks:
        content = c.get("content")
        if content:
            lines.append(f"- {content}")
    
    return "\n".join(lines) if lines else "No specific SEC section context found."


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Cache keys
    "normalize_symbol",
    "ck_fund",
    "ck_tav",
    "ck_task",
    "ck_report",
    "ck_tech",
    
    # Data fetching
    "fetch_history_points",
    "get_fundamentals_with_cache",
    "get_news_with_optional_tavily_fallback",
    "get_sec_routed_context",
    
    # Analysis
    "compute_market_snapshot",
    "compute_data_quality",
    "build_technicals_text_from_pack",
    
    # Validation
    "validate_report",
    
    # Utilities
    "json_dumps",
    "stable_hash",
    "preview",
    "shorten_text",
    "db_session",
    
    # Trimming
    "trim_profile",
    "trim_quote",
    "trim_market_snapshot",
    "trim_earnings_calendar",
    "trim_sec_chunks",
    "trim_news_items",
]
