from decimal import Decimal
from typing import Any, Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
from models.holding import Holding, HoldingOut
from services.finnhub_service import FinnhubService
from math import fsum
import time

def _to_float(x: Any) -> float:
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return 0.0

def _key(symbol: Optional[str], typ: Optional[str]) -> str:
    s = (symbol or "").upper().strip()
    t = (typ or "").lower().strip()
    return f"{s}:{t}" if t else s

def _position_value(h: HoldingOut) -> float:
    # Prefer explicit total value, fall back to price * qty
    v = _to_float(getattr(h, "value", 0.0) or getattr(h, "current_value", 0.0))
    if v > 0:
        return v
    return _to_float(h.current_price) * _to_float(h.quantity)

def _enrich_pl_fields(h: HoldingOut) -> HoldingOut:
    out = h.model_copy(deep=True)

    qty  = _to_float(getattr(h, "quantity", 0.0))
    curr = getattr(h, "current_price", None)
    # Prefer explicit unit cost if present, else legacy purchase_price
    pur  = getattr(h, "purchase_unit_price", None) or getattr(h, "purchase_price", None)
    pc   = getattr(h, "previous_close", None)

    curr_f = _to_float(curr) if curr is not None else None
    pur_f  = _to_float(pur)  if pur  is not None else None
    pc_f   = _to_float(pc)   if pc   is not None else None

    # Intraday P/L
    out.day_pl = None if (curr_f is None or pc_f is None or pc_f <= 0) \
        else round((curr_f - pc_f) * qty, 2)

    # Total unrealized P/L
    out.unrealized_pl = None if (curr_f is None or pur_f is None or pur_f <= 0) \
        else round((curr_f - pur_f) * qty, 2)

    return out

def get_all_holdings(user_id: str, db: Session) -> List[Holding]:
    return db.query(Holding).filter_by(user_id=user_id).all()

async def get_holdings_with_live_prices_top(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
    *,
    top_only: bool = False,
    top_n: int = 5,
    include_weights: bool = True,
) -> Dict[str, Any]:
    rows = get_all_holdings(user_id, db)
    if not rows:
        return {
            "items": [],
            "as_of": int(time.time()),
            "price_status": "live",
            "requested_currency": currency.upper(),
        }

    # 1) Request and normalize quote keys to our _key format
    pairs: List[Tuple[str, str]] = [
        ((h.symbol or ""), (h.type or "")) for h in rows if (h.symbol or "").strip()
    ]
    quotes_raw = await finnhub.get_prices(pairs=pairs, currency=currency)

    normalized_quotes: Dict[str, Dict[str, Any]] = {}
    for k, v in (quotes_raw or {}).items():
        if isinstance(k, (tuple, list)):
            kk = _key(k[0] if k else None, (k[1] if len(k) > 1 else None))
        else:
            kk = _key(str(k), None)
        if v and v.get("currentPrice") not in (None, 0):
            normalized_quotes[kk] = v

    # 2) Build enriched items (current_price / previous_close / value / price_status)
    items = enrich_holdings(rows, normalized_quotes, currency)

    # compute top regardless of top_only
    market_value = sum(max(_position_value(it), 0.0) for it in items)
    sorted_items = sorted(items, key=_position_value, reverse=True)

    top_items: List[HoldingOut] = []
    for it in sorted_items[: max(1, top_n)]:
        v = _position_value(it)
        copy = _enrich_pl_fields(it)      # fills day_pl & unrealized_pl
        copy.value = round(v, 2)
        if include_weights:
            copy.weight = round(v / market_value * 100.0, 2) if market_value > 0 else None
        top_items.append(copy)

    payload = {
        "items": items if not top_only else top_items,   # <— when top_only, return only top
        "top_items": top_items,                           # <— always include enriched top slice
        "as_of": int(time.time()),
        "price_status": "live",
        "requested_currency": currency.upper(),
    }
    return payload

async def get_holdings_with_live_prices(
    user_id: str,
    db: Session,
    finnhub: FinnhubService,
    currency: str = "USD",
) -> Dict[str, Any]:
    rows = get_all_holdings(user_id, db)
    if not rows:
        return {"items": [], "as_of": int(time.time()), "price_status": "live", "requested_currency": currency.upper()}

    pairs: List[Tuple[str, str]] = [(h.symbol or "", h.type or "") for h in rows if h.symbol and h.symbol.strip()]
    quotes = await finnhub.get_prices(pairs=pairs, currency=currency)
    filtered_quotes = {k: v for k, v in quotes.items() if v and v.get("currentPrice") is not None and v.get("currentPrice") != 0}

    items = enrich_holdings(rows, filtered_quotes, currency)
    return {
        "items": items,
        "as_of": int(time.time()),
        "price_status": "live",
        "requested_currency": currency.upper(),
    }

def enrich_holdings(
    holdings: List[Holding],
    prices: Dict[str, Dict[str, Any]],
    currency: str = "USD",
) -> List[HoldingOut]:
    enriched: List[HoldingOut] = []
    for h in holdings:
        k = _key(h.symbol, h.type)
        q = prices.get(k) or prices.get(_key(h.symbol, None))  # fallback to symbol-only

        qty = _to_float(h.quantity)

        dto = HoldingOut(
            id=h.id,
            user_id=h.user_id,
            source=h.source,
            external_id=h.external_id,
            symbol=h.symbol,
            name=h.name,
            type=h.type,
            quantity=h.quantity,
            current_price=h.current_price,
            purchase_price=h.purchase_price,
            value=h.value,
            account_name=h.account_name,
            institution=h.institution,
            currency=h.currency,
            purchase_amount_total=getattr(h, "purchase_amount_total", None),
            purchase_unit_price=getattr(h, "purchase_unit_price", None),
            unrealized_pl=getattr(h, "unrealized_pl", None),
            unrealized_pl_pct=getattr(h, "unrealized_pl_pct", None),
            current_value=getattr(h, "current_value", None),
        )

        if q and q.get("currentPrice") is not None:
            live_price = _to_float(q["currentPrice"])
            dto.current_price = live_price
            dto.previous_close = (
                _to_float(q.get("previousClose")) if q.get("previousClose") is not None else None
            )
            dto.currency = q.get("currency") or (dto.currency or currency.upper())
            dto.value = round(live_price * qty, 2)
            dto.price_status = "live"
        else:
            dto.price_status = "unavailable"
            if dto.current_price is not None:
                try:
                    dto.value = round(_to_float(dto.current_price) * qty, 2)
                except Exception:
                    pass

        enriched.append(dto)

    return enriched
