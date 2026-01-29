# services/ai/portfolio/types.py
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


class HoldingRole(str, Enum):
    core = "core"              # main drivers of portfolio returns
    risk_amplifier = "risk_amplifier"  # adds downside/volatility risk
    satellite = "satellite"    # small positions / doesn’t move portfolio much
    unknown = "unknown"        # fallback if data is missing


class HoldingFlag(str, Enum):
    concentration = "concentration"                # big weight
    big_loser = "big_loser"                        # large unrealized loss %
    high_volatility_type = "high_volatility_type"  # crypto or similar
    missing_cost_basis = "missing_cost_basis"      # no purchase totals / unit price
    missing_price = "missing_price"                # current price or value missing/zero
    tiny_position = "tiny_position"                # very small weight
    data_quality = "data_quality"                  # generic “watch data” flag


class HoldingClassification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    symbol: str
    type: Optional[str] = None

    value: Optional[float] = None
    weight: Optional[float] = None
    unrealized_pl: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None

    # ✅ ADD THESE
    is_driver: bool = False
    is_risk_amplifier: bool = False

    flags: List[HoldingFlag] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    score: float = 0.0



class ClassifyConfig(BaseModel):
    """
    Tune these without touching logic.
    You can store defaults in env/config later.
    """
    top_n_core: int = 5
    core_weight_pct: float = 10.0          # >= 10% => core
    core_value_share_pct: float = 0.0      # optional: if you want a floor by value share
    big_loser_pct: float = -15.0           # <= -15% unrealized => risk amplifier
    tiny_weight_pct: float = 1.0           # <= 1% => tiny position flag
    crypto_types: Tuple[str, ...] = ("crypto",)
    risk_types: Tuple[str, ...] = ("crypto",)  # treat these as high-volatility by default