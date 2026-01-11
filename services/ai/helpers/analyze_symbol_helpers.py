from typing import Any, Dict, List
from utils.common_helpers import safe_float
import asyncio
from services.yahoo_service import get_price_history

def preview(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")

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
