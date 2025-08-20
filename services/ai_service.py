import json
import math
import datetime as dt
from typing import Any, Dict, List, TypedDict, Optional

from langchain_openai import ChatOpenAI  # pip install langchain-openai
from services.yahoo_service import get_full_stock_data
from types.holding import HoldingInput

# Deterministic + bounded
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o-mini",   # pick your model; use something inexpensive+fast
    max_completion_tokens=700,
    timeout=20,            # seconds
)

# ---------- Types ----------

class AnalysisOutput(TypedDict, total=False):
    symbol: str
    as_of_utc: str
    pnl_abs: float
    pnl_pct: float
    market_context: Dict[str, Any]
    rating: str                 # "hold" | "sell" | "watch" | "diversify"
    rationale: str
    key_risks: List[str]
    suggestions: List[str]
    data_notes: List[str]
    disclaimer: str

# ---------- Helpers ----------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _pct_change(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base in (None, 0):
        return None
    return (cur / base - 1.0) * 100.0

def _round(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), d)

# ---------- Core ----------
def analyze_investment_holding(holding: HoldingInput) -> AnalysisOutput:
    """
    Deterministic, structured analysis for a single holding.
    - Computes numbers in Python
    - Uses Yahoo data (services.yahoo_service.get_full_stock_data)
    - Asks the LLM only to EXPLAIN, with a fixed JSON schema
    """
    symbol = holding["symbol"].upper().strip()
    yh = get_full_stock_data(symbol, include_news=False)

    if yh.get("status") != "ok":
        # Surface a minimal, consistent error object
        return {
            "symbol": symbol,
            "as_of_utc": dt.datetime.utcnow().isoformat(),
            "rating": "watch",
            "rationale": f"Could not fetch quote data for {symbol}. Reason: {yh.get('message') or 'unknown error'}.",
            "key_risks": [],
            "suggestions": [],
            "data_notes": [f"Data fetch error: {yh.get('error_code')}", "Source: Yahoo Finance via yahooquery"],
            "disclaimer": "This is educational information, not investment advice.",
        }

    # Prefer Yahoo’s current_price over provided, but keep user-provided as fallback
    current_price = _safe_float(yh.get("current_price")) or _safe_float(holding.get("current_price"))
    purchase_price = _safe_float(holding.get("purchase_price"))
    quantity = _safe_float(holding.get("quantity")) or 0.0
    currency = holding.get("currency") or yh.get("currency") or "USD"

    # Portfolio math
    cost_basis = (purchase_price or 0.0) * quantity
    market_value = (current_price or 0.0) * quantity
    pnl_abs = market_value - cost_basis
    pnl_pct = _pct_change(market_value, cost_basis)

    # Price context
    prev_close = _safe_float(yh.get("previous_close"))
    day_change = None if (current_price is None or prev_close is None) else (current_price - prev_close)
    day_change_pct = _pct_change(current_price, prev_close)
    dist_from_high_pct = _safe_float(yh.get("distance_from_52w_high_pct"))
    dist_from_low_pct = _safe_float(yh.get("distance_from_52w_low_pct"))

    # Fundamentals context (may be None for ETFs/crypto)
    pe_ratio = _safe_float(yh.get("pe_ratio"))
    forward_pe = _safe_float(yh.get("forward_pe"))
    price_to_book = _safe_float(yh.get("price_to_book"))
    dividend_yield = _safe_float(yh.get("dividend_yield"))
    beta = _safe_float(yh.get("beta"))

    # Build compact context for the model (ONLY what it should read)
    market_context = {
        "current_price": _round(current_price, 4),
        "previous_close": _round(prev_close, 4),
        "day_change": _round(day_change, 4),
        "day_change_pct": _round(day_change_pct, 4),
        "52_week_high": _round(_safe_float(yh.get("52_week_high")), 4),
        "52_week_low": _round(_safe_float(yh.get("52_week_low")), 4),
        "distance_from_52w_high_pct": _round(dist_from_high_pct, 4),
        "distance_from_52w_low_pct": _round(dist_from_low_pct, 4),
        "pe_ratio": _round(pe_ratio, 4),
        "forward_pe": _round(forward_pe, 4),
        "price_to_book": _round(price_to_book, 4),
        "dividend_yield": _round(dividend_yield, 4),
        "beta": _round(beta, 4),
        "exchange": yh.get("exchange"),
        "currency": currency,
    }

    # Notes for transparency
    data_notes = [
        f"Quote time (UTC): {yh.get('quote_time_utc') or 'N/A'}",
        f"Currency: {currency}",
        f"Source: Yahoo Finance via yahooquery",
    ]
    dq = yh.get("data_quality") or {}
    if dq.get("is_stale") is True:
        data_notes.append("Quote may be stale")
    missing = dq.get("missing_fields") or []
    if missing:
        data_notes.append(f"Missing fields: {', '.join(missing)}")

    # Strong guardrails in system prompt
    system = (
        "You are a cautious retail investing explainer. "
        "Use ONLY the provided fields; do not invent numbers, do not forecast prices, do not give personalized financial advice. "
        "Write clearly in plain language. Keep it concise and specific. "
        "You must return VALID JSON only, matching the schema I provide."
    )

    # Tailor advice style by asset type
    holding_type = (holding.get("type") or "stock").strip().lower()
    style_hint = {
        "stock": "Focus on valuation (P/E, P/B, yield if available), volatility (beta), and company vs market drivers.",
        "etf": "Focus on country/sector concentration, currency risk, expense ratio (if known), and tracking vs the market.",
        "crypto": "Emphasize high volatility, regulatory/market structure risks, and position sizing.",
    }.get(holding_type, "Focus on the most relevant risks given the instrument type.")

    # Strict output schema (text form—model must conform)
    schema_text = {
        "symbol": "string",
        "as_of_utc": "ISO-8601 string",
        "pnl_abs": "number",
        "pnl_pct": "number",
        "market_context": "object with keys current_price, previous_close, day_change, day_change_pct, 52_week_high, 52_week_low, distance_from_52w_high_pct, distance_from_52w_low_pct, pe_ratio, forward_pe, price_to_book, dividend_yield, beta, exchange, currency",
        "rating": "one of: hold | sell | watch | diversify",
        "rationale": "short paragraph explaining reasoning using provided fields only",
        "key_risks": "array of 3-5 short bullet points",
        "suggestions": "array of 2-4 short, actionable tips (e.g., diversification, position sizing, risk management)",
        "data_notes": "array of short strings about data limitations and provenance",
        "disclaimer": "string: 'This is educational information, not financial advice.'",
    }

    user_payload = {
        "holding": {
            "symbol": symbol,
            "name": holding.get("name") or yh.get("name"),
            "type": holding_type,
            "quantity": quantity,
            "purchase_price": purchase_price,
            "institution": holding.get("institution", "N/A"),
            "currency": currency,
        },
        "computed": {
            "cost_basis": _round(cost_basis, 4),
            "market_value": _round(market_value, 4),
            "pnl_abs": _round(pnl_abs, 4),
            "pnl_pct": _round(pnl_pct, 4),
        },
        "market_context": market_context,
        "style_hint": style_hint,
        "must_include": [
            "Say clearly if the user is up or down (and by how much).",
            "Use 52-week context if present.",
            "For ETFs, emphasize country/sector concentration and currency risk.",
            "Give a single-word rating: hold, sell, watch, or diversify.",
            "Include data notes and the standard disclaimer.",
        ],
        "output_schema": schema_text,
        "disclaimer": "This is educational information, not financial advice.",
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload)},
    ]

    ai = llm.invoke(messages)

    # Parse + safe fallback
    try:
        # obj = json.loads(ai.content)
        raw_content = ai.content
        if isinstance(raw_content, list):
    # flatten pieces into one string
            raw_content = "".join(str(p) for p in raw_content)

        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
            
        obj = json.loads(raw_content)
    except Exception:
        # Fallback: wrap as structured object with the model text as rationale
        obj = {
            "symbol": symbol,
            "as_of_utc": dt.datetime.utcnow().isoformat(),
            "pnl_abs": _round(pnl_abs, 4),
            "pnl_pct": _round(pnl_pct, 4),
            "market_context": market_context,
            "rating": "watch",
            "rationale": ai.content[:1200],
            "key_risks": [],
            "suggestions": [],
            "data_notes": data_notes,
            "disclaimer": "This is educational information, not financial advice.",
        }

    # Ensure required fields exist + fill with our computed numbers
    obj.setdefault("symbol", symbol)
    obj.setdefault("as_of_utc", dt.datetime.utcnow().isoformat())
    obj.setdefault("pnl_abs", _round(pnl_abs, 4))
    obj.setdefault("pnl_pct", _round(pnl_pct, 4))
    obj["market_context"] = market_context  # overwrite with trusted numbers
    obj.setdefault("data_notes", data_notes)
    obj.setdefault("disclaimer", "This is educational information, not financial advice.")

    # Optional: keep rating sane
    valid_ratings = {"hold", "sell", "watch", "diversify"}
    rating = str(obj.get("rating", "watch")).lower().strip()
    obj["rating"] = rating if rating in valid_ratings else "watch"

    return obj  # type: ignore[return-value]
