# routers/holdings_routes.py
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

import schemas.general as general
from database import get_db
from models.holding import Holding
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService, format_finnhub_symbol
from services.holding_service import get_all_holdings, get_holdings_broker_only, create_holding, update_holding
from services.currency_service import resolve_currency, fx_pair_rate
from services.supabase_auth import get_current_db_user
from services.yahoo_service import get_price_history
from utils.common_helpers import to_float, canonical_key, normalize_asset_type

router = APIRouter()

def _yahoo_symbol(symbol: str, typ: str, currency: str) -> str:
    """Yahoo Finance symbol: CAD equity/ETF use .TO; crypto use SYMBOL-USD."""
    s = (symbol or "").upper().strip()
    t = (typ or "").lower().strip()
    ccy = (currency or "").upper().strip()
    if not s:
        return s
    if t in ("cryptocurrency", "crypto"):
        return s if "-" in s else f"{s}-USD"
    if ccy == "CAD" and t in ("equity", "etf", "stock") and "." not in s:
        return f"{s}.TO"
    return s

@router.post("/holdings")
def save_holding(
    holding: general.HoldingCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    return create_holding(
        db,
        user.id,
        holding.symbol,
        holding.quantity,
        holding.purchase_price,
        holding.type,
        name=holding.name,
        currency=holding.currency or "USD",
    )

@router.put("/holdings/{holding_id}")
def edit_holding(
    holding_id: int,
    payload: general.HoldingUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    try:
        updates = payload.model_dump(exclude_none=True)
        updated = update_holding(db, user.id, holding_id, updates)
        return updated
    except ValueError:
        raise HTTPException(status_code=404, detail="Holding not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Only manually added holdings can be edited")

@router.get("/holdings")
async def get_holdings(
    includePrices: bool = Query(False),
    currency: str | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    """Holdings from broker (DB) only. Total cost is converted to the user's chosen currency (settings).
    When includePrices=true, current_price is filled from Finnhub (Canadian equity/ETF use TSX .TO for CAD;
    crypto quotes are in USD and are converted to the holding's currency when different)."""
    if not includePrices:
        return get_all_holdings(str(user.id), db)

    data = get_holdings_broker_only(str(user.id), db)
    target_currency = resolve_currency(user, currency)
    if not data.get("items"):
        data["currency"] = target_currency
        return data

    # Build (symbol, type) pairs for Finnhub; use TSX .TO for CAD equity/ETF so prices are in CAD
    pairs: list[tuple[str, str]] = []
    for it in data["items"]:
        sym = (getattr(it, "symbol", None) or "").strip()
        typ = getattr(it, "type", None) or ""
        ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        if not sym:
            continue
        fs = format_finnhub_symbol(sym, typ, ccy)
        typ_n = normalize_asset_type(typ) or "equity"
        pairs.append((fs, typ_n))

    if pairs:
        quotes = await finnhub.get_prices_cached(pairs)
        for it in data["items"]:
            sym = (getattr(it, "symbol", None) or "").strip()
            typ = getattr(it, "type", None) or ""
            ccy = (getattr(it, "currency", None) or "USD").strip().upper()
            if not sym:
                continue
            fs = format_finnhub_symbol(sym, typ, ccy)
            typ_n = normalize_asset_type(typ) or "equity"
            key = canonical_key(fs, typ_n)
            q = quotes.get(key)
            if isinstance(q, dict) and q.get("currentPrice") is not None:
                price = float(q["currentPrice"])
                # Finnhub crypto quotes are always in USD; convert to holding currency if needed
                if typ_n == "cryptocurrency" and ccy != "USD":
                    rate = await fx_pair_rate("USD", ccy)
                    price = round(price * rate, 8)
                it.current_price = price

    total_in_target = 0.0
    for it in data["items"]:
        from_ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        rate = await fx_pair_rate(from_ccy, target_currency)
        cost_in_holding_ccy = to_float(getattr(it, "value", 0))
        converted = round(cost_in_holding_ccy * rate, 2)
        it.value_in_display_currency = converted
        total_in_target += converted
    data["market_value"] = round(total_in_target, 2)
    data["currency"] = target_currency
    return data


def _close_for_date(points: list, date_ts: int) -> float | None:
    """Return close price for the given day (UTC date as start-of-day epoch). Forward-fill if no point that day."""
    if not points:
        return None
    day_end = date_ts + 86400
    same_day = [p for p in points if isinstance(p, dict) and date_ts <= p.get("t", 0) < day_end]
    if same_day:
        c = same_day[-1].get("c")
        return float(c) if c is not None and (isinstance(c, (int, float))) else None
    # forward-fill: latest point before this date
    before = [p for p in points if isinstance(p, dict) and p.get("t", 0) < date_ts]
    if not before:
        return None
    last = max(before, key=lambda p: p.get("t", 0))
    c = last.get("c")
    return float(c) if c is not None and isinstance(c, (int, float)) else None


@router.get("/holdings/portfolio-history")
async def get_portfolio_history(
    days: int = Query(7, ge=1, le=30),
    currency: str | None = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    """Last N days of portfolio value (sum of position values from Yahoo history), in user's display currency."""
    data = get_holdings_broker_only(str(user.id), db)
    target_currency = resolve_currency(user, currency)
    items = data.get("items") or []
    if not items:
        return {"points": [], "currency": target_currency}

    # Fetch history once per unique Yahoo symbol
    cache: dict[str, list] = {}
    for it in items:
        sym = (getattr(it, "symbol", None) or "").strip()
        typ = getattr(it, "type", None) or ""
        ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        if not sym:
            continue
        ysym = _yahoo_symbol(sym, typ, ccy)
        if ysym not in cache:
            raw = await run_in_threadpool(
                get_price_history, ysym, "1mo", "1d"
            )
            pts = raw.get("points", []) if isinstance(raw, dict) and raw.get("status") == "ok" else []
            cache[ysym] = sorted(pts, key=lambda p: p.get("t", 0))
        else:
            pts = cache[ysym]

    # Unique (yahoo_symbol, quantity, currency) per row so we can value each position
    positions: list[tuple[str, float, str]] = []
    for it in items:
        sym = (getattr(it, "symbol", None) or "").strip()
        typ = getattr(it, "type", None) or ""
        ccy = (getattr(it, "currency", None) or "USD").strip().upper()
        qty = to_float(getattr(it, "quantity", 0))
        if not sym or qty <= 0:
            continue
        ysym = _yahoo_symbol(sym, typ, ccy)
        positions.append((ysym, qty, ccy))

    # Last N calendar days (start of day UTC)
    now = int(time.time())
    day_ts_list = [now - (i * 86400) for i in range(days)]
    day_ts_list = [ts - (ts % 86400) for ts in day_ts_list]
    day_ts_list = sorted(set(day_ts_list))[-days:]

    out_points: list[dict] = []
    for date_ts in day_ts_list:
        total = 0.0
        for ysym, qty, ccy in positions:
            pts = cache.get(ysym, [])
            close = _close_for_date(pts, date_ts)
            if close is None:
                continue
            value_ccy = qty * close
            rate = await fx_pair_rate(ccy, target_currency)
            total += value_ccy * rate
        out_points.append({"t": date_ts, "value": round(total, 2)})

    out_points.sort(key=lambda x: x["t"])
    return {"points": out_points, "currency": target_currency}


@router.delete("/holdings/{holding_id}")
def delete_holding( 
    holding_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_db_user),
):
    holding = db.query(Holding).filter_by(id=holding_id, user_id=user.id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")

    db.delete(holding)
    db.commit()
    return {"detail": "Deleted"}
