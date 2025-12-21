from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from models.user import User
import time
import httpx

SUPPORTED_CCY = {"USD", "CAD"}
ALLOWED_TYPES = {"equity", "etf", "cryptocurrency"}

def normalize_currency(iso_code: Optional[str], unofficial_code: Optional[str] = None) -> str:
    ccy = (iso_code or unofficial_code or "USD").upper()
    return ccy if ccy in SUPPORTED_CCY else "USD"


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

# ---------- FX ----------
async def get_usd_to_cad_rate(self, client: Optional[httpx.AsyncClient] = None) -> float:
    now = time.time()
    if self._fx_cache and self._fx_cache[1] > now:
        return self._fx_cache[0]

    async with self._client(client) as c:
        try:
            r = await c.get("https://api.frankfurter.app/latest?from=USD&to=CAD")
            data = r.json()
            rate = data.get("rates", {}).get("CAD")
            if rate is None:
                print("⚠️ FX fetch failed, defaulting to 1.0: rate missing")
                return 1.0
            rate = float(rate)
            if self._fx_ttl:
                self._fx_cache = (rate, now + self._fx_ttl)
            return rate
        except Exception as e:
            # If FX fails, fall back to 1.0 rather than exploding the whole request.
            # You can choose to re-raise if you want hard failures.
            print("⚠️ FX fetch failed, defaulting to 1.0:", e)
            return 1.0