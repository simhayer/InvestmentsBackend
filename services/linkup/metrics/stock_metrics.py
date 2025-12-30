from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Sequence

from services.currency_service import fx_pair_rate
from services.linkup.metrics.build_metrics_yahoo import build_metrics, holdings_to_positions
from services.yahoo_service import get_overview


class StockMetricsCalculator:
    def __init__(self, base_currency: str = "USD") -> None:
        self.base_currency = (base_currency or "USD").upper()

    async def build_for_symbol(
        self,
        symbol: str,
        holdings: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        symbol_up = (symbol or "").upper().strip()
        if not symbol_up or not holdings:
            return {}

        positions = holdings_to_positions(list(holdings), base_currency=self.base_currency)
        if not positions:
            return {}

        metrics = await build_metrics(positions, base_currency=self.base_currency)
        per_symbol = metrics.get("per_symbol", {}).get(symbol_up)
        if not per_symbol:
            return {}

        overview = get_overview(symbol_up)
        if isinstance(overview, dict) and overview.get("status") == "ok":
            sector = overview.get("sector")
            country = overview.get("country")
            if sector:
                per_symbol["sector"] = sector
            if country:
                per_symbol["region"] = country

        total_qty = sum(
            float(p.quantity or 0.0)
            for p in positions
            if (p.symbol or "").upper() == symbol_up
        )
        total_cost = sum(
            float(p.cost_basis_total or 0.0)
            for p in positions
            if (p.symbol or "").upper() == symbol_up
        )
        if total_qty:
            per_symbol["quantity"] = round(total_qty, 8)
        if total_cost:
            per_symbol["cost_basis_total"] = round(total_cost, 8)

        fx_info = await self._build_fx_info(per_symbol.get("quote_currency"))
        if fx_info:
            per_symbol["fx_exposure"] = fx_info

        return {
            "position_metrics": per_symbol,
            "portfolio_context": metrics.get("portfolio", {}),
            "fx_exposure": fx_info or {},
        }

    async def _build_fx_info(self, quote_currency: Optional[str]) -> Dict[str, Any]:
        quote = (quote_currency or "").upper()
        base = self.base_currency
        if not quote:
            return {}
        rate = await fx_pair_rate(quote, base)
        return {
            "quote_currency": quote,
            "base_currency": base,
            "fx_rate": float(rate),
            "fx_required": quote != base,
        }

    def build_for_symbol_sync(
        self,
        symbol: str,
        holdings: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return {}

        return asyncio.run(self.build_for_symbol(symbol, holdings))
