# metrics/positions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Position:
    """
    Canonical analytics input type (source-agnostic).
    Everything in the analytics pipeline should depend on THIS (not Plaid, not DTOs).
    """
    symbol: str
    quantity: float
    cost_basis_total: float  # total cost in base_currency
    name: Optional[str] = None
    asset_class: str = "other"  # equity/etf/cryptocurrency/cash/other
    currency: str = "USD"       # informational; analytics converts to base_currency where needed
