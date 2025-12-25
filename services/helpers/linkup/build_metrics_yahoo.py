from __future__ import annotations

from typing import Any, Dict, List, Tuple

from services.helpers.linkup.position import Position
from services.yahoo_service import get_full_stock_data_many
from services.currency_service import get_usd_to_cad_rate, fx_pair_rate

def normalize_symbol(sym: str, asset_class: str, base_currency: str) -> str:
    sym = sym.strip().upper()
    if asset_class == "cryptocurrency" and "-" not in sym:
        quote = "USD" if base_currency.upper() == "USD" else "CAD"
        return f"{sym}-{quote}"
    return sym

def holdings_to_positions(holdings: List[Any], *, base_currency: str) -> List[Position]:
    """
    HoldingOut -> Position (canonical analytics input)
    """
    out: List[Position] = []
    base_currency = (base_currency or "USD").upper()

    for h in holdings:
        sym = normalize_symbol((getattr(h, "symbol", None) or "").strip().upper(), (getattr(h, "type", None) or "other").lower(), base_currency)
        if not sym:
            continue

        qty = float(getattr(h, "quantity", None) or 0.0)

        # TOTAL cost basis priority
        cb = float(getattr(h, "purchase_amount_total", None) or 0.0)
        if cb <= 0:
            unit = float(getattr(h, "purchase_unit_price", None) or 0.0) or float(getattr(h, "purchase_price", None) or 0.0)
            cb = unit * qty

        out.append(
            Position(
                symbol=sym,
                quantity=qty,
                cost_basis_total=float(cb),
                name=(getattr(h, "name", None) or sym),
                asset_class=(getattr(h, "type", None) or "other").lower(),
                currency=(getattr(h, "currency", None) or base_currency).upper(),
            )
        )

    return out

async def build_metrics(
    positions: List[Position],
    *,
    base_currency: str = "USD",
) -> Dict[str, Any]:
    """
    Portfolio metrics using Yahoo quotes/fundamentals.
    - Uses USD/CAD FX only (via get_usd_to_cad_rate)
    - Returns the same overall structure your pipeline expects:
        { "per_symbol": {...}, "portfolio": {...} }
    """
    base_currency = (base_currency or "USD").upper()

    # Merge duplicate symbols (multiple accounts)
    merged: Dict[str, Position] = {}
    for p in positions:
        sym = (p.symbol or "").strip().upper()
        if not sym:
            continue
        if sym not in merged:
            merged[sym] = Position(
                symbol=sym,
                quantity=float(p.quantity or 0.0),
                cost_basis_total=float(p.cost_basis_total or 0.0),
                name=p.name or sym,
                asset_class=(p.asset_class or "other").lower(),
                currency=(p.currency or base_currency).upper(),
            )
        else:
            prev = merged[sym]
            merged[sym] = Position(
                symbol=sym,
                quantity=float(prev.quantity) + float(p.quantity or 0.0),
                cost_basis_total=float(prev.cost_basis_total) + float(p.cost_basis_total or 0.0),
                name=prev.name or p.name or sym,
                asset_class=prev.asset_class or (p.asset_class or "other").lower(),
                currency=prev.currency or (p.currency or base_currency).upper(),
            )

    positions = list(merged.values())
    if not positions:
        return {
            "per_symbol": {},
            "portfolio": {
                "total_value": 0.0,
                "cash_value": 0.0,
                "num_positions": 0,
                "concentration_top_5_pct": 0.0,
                "asset_class_weights_pct": {},
            },
        }

    symbols = [p.symbol for p in positions]
    quotes = get_full_stock_data_many(symbols)  # {SYM: {status, current_price, currency, ...}}

    usd_to_cad = await get_usd_to_cad_rate()

    per_symbol: Dict[str, Any] = {}
    total_value = 0.0
    cash_value = 0.0

    # First pass: compute market values
    mv_map: Dict[str, float] = {}

    for p in positions:
        q = quotes.get(p.symbol) or {}
        q_status = q.get("status")
        q_cur = (q.get("currency") or p.currency or base_currency).upper()

        cur_px = q.get("current_price")
        try:
            cur_px_f = float(cur_px) if cur_px is not None else 0.0
        except Exception:
            cur_px_f = 0.0

        fx = await fx_pair_rate(q_cur, base_currency)
        px_base = cur_px_f * fx
        mv = px_base * float(p.quantity or 0.0)

        mv = round(float(mv), 8)
        mv_map[p.symbol] = mv
        total_value += mv

        if (p.asset_class or "").lower() == "cash" or p.symbol == "CASH":
            cash_value += mv

        cb = float(p.cost_basis_total or 0.0)
        unreal_abs = (mv - cb) if cb > 0 else None
        unreal_pct = ((unreal_abs / cb) * 100.0) if (cb > 0 and unreal_abs is not None) else None

        per_symbol[p.symbol] = {
            "symbol": p.symbol,
            "name": p.name or q.get("name") or p.symbol,
            "asset_class": (p.asset_class or "other").lower(),
            "sector": "",
            "region": "",
            "weight_pct": 0.0,  # fill after totals
            "market_value": mv,
            "cost_basis": round(cb, 8),
            "unrealized_pnl_abs": None if unreal_abs is None else round(float(unreal_abs), 8),
            "unrealized_pnl_pct": None if unreal_pct is None else round(float(unreal_pct), 8),

            # optional “fundamentals” (handy for AI + UI)
            "beta_1Y": q.get("beta"),
            "pe_ratio": q.get("pe_ratio"),
            "forward_pe": q.get("forward_pe"),
            "market_cap": q.get("market_cap"),
            "dividend_yield": q.get("dividend_yield"),
            "price_to_book": q.get("price_to_book"),

            "quote_currency": q_cur,
            "price_status": "live" if q_status == "ok" else "missing",
        }

    # Weights
    if total_value > 0:
        for sym, rec in per_symbol.items():
            rec["weight_pct"] = round(100.0 * float(rec["market_value"]) / float(total_value), 2)

    # Concentration top 5
    weights_sorted = sorted([float(r["weight_pct"]) for r in per_symbol.values()], reverse=True)
    concentration_top_5_pct = round(sum(weights_sorted[:5]), 8) if weights_sorted else 0.0

    # Asset class weights
    asset_class_weights: Dict[str, float] = {}
    for r in per_symbol.values():
        ac = r.get("asset_class") or "other"
        asset_class_weights[ac] = asset_class_weights.get(ac, 0.0) + float(r.get("weight_pct") or 0.0)

    portfolio = {
        "total_value": round(float(total_value), 2),
        "cash_value": round(float(cash_value), 2),
        "num_positions": len(per_symbol),
        "concentration_top_5_pct": concentration_top_5_pct,
        "asset_class_weights_pct": {k: round(v, 8) for k, v in asset_class_weights.items()},
        "base_currency": base_currency,
        "usd_to_cad": float(usd_to_cad),
    }

    return {"per_symbol": per_symbol, "portfolio": portfolio}
