import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from models.user import User
import time
import httpx

logger = logging.getLogger(__name__)

def infer_base_currency_from_holdings(plaid_holdings: List[Dict]) -> str:
    """
    Infer base currency using value-weighted totals.
    If totals tie or no values, default USD.
    """
    cad_total = 0.0
    usd_total = 0.0

    for h in plaid_holdings:
        ccy = (h.get("currency") or "").upper()
        val = h.get("currentValue")
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue

        if ccy == "CAD":
            cad_total += v
        elif ccy == "USD":
            usd_total += v

    return "CAD" if cad_total > usd_total else "USD"


def maybe_auto_set_user_base_currency(
    user: User,
    normalized_holdings: List[Dict],
    db: Session,
) -> None:
    """
    Auto-set user's base currency ONLY if they haven't chosen manually.
    Uses holdings to infer.
    """
    # If user manually chose, never override
    if getattr(user, "base_currency_source", "default") == "manual":
        return

    inferred = infer_base_currency_from_holdings(normalized_holdings)

    # Only update if it actually changes something or if source wasn't auto yet
    user.currency = inferred
    user.base_currency_source = "auto"
    db.add(user)
    db.commit()
    db.refresh(user)

# # ---------- FX ----------
# --- Module-level TTL cache ---
_FX_TTL_SEC = 60
_fx_cache: Optional[tuple[float, float]] = None  # (rate, expires_at)
async def get_usd_to_cad_rate() -> float:
    """
    Fetch USD->CAD FX rate with a small in-memory TTL cache.
    Owns its own httpx client (no external client dependency).
    Falls back to 1.0 on any failure.
    """
    global _fx_cache

    now = time.time()
    if _fx_cache and _fx_cache[1] > now:
        return _fx_cache[0]

    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get("https://api.frankfurter.app/latest?from=USD&to=CAD")
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("FX fetch failed, defaulting to 1.0: %s", e)
        return 1.0
    try:
        rate = data.get("rates", {}).get("CAD")
        if rate is None:
            logger.warning("FX fetch failed, defaulting to 1.0: rate missing")
            return 1.0

        rate_f = float(rate)
        if rate_f <= 0:
            logger.warning("FX fetch failed, defaulting to 1.0: invalid rate")
            return 1.0

        _fx_cache = (rate_f, now + _FX_TTL_SEC)
        return rate_f
    except Exception as e:
        logger.warning("FX parse failed, defaulting to 1.0: %s", e)
        return 1.0
    
def resolve_currency(user: User, currency_query: str | None) -> str:
    if currency_query and currency_query.strip():
        return currency_query.strip().upper()

    base = getattr(user, "currency", None)
    if base and str(base).strip():
        return str(base).strip().upper()

    return "USD"

async def fx_pair_rate(from_cur: str, to_cur: str) -> float:
    """
    USD/CAD only:
      - USD->CAD = usd_to_cad
      - CAD->USD = 1/usd_to_cad
      - same currency = 1
    """
    a = (from_cur or "").upper()
    b = (to_cur or "").upper()
    if a == b:
        return 1.0
    if a == "USD" and b == "CAD":
        return float(await get_usd_to_cad_rate())
    if a == "CAD" and b == "USD":
        rate = await get_usd_to_cad_rate()
        return float(1.0 / rate) if rate else 1.0
    # if something unexpected shows up, don’t break math
    return 1.0
