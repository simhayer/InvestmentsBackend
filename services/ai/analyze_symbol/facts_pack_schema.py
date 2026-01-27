from typing import Dict, List, Literal, Optional
from pydantic import BaseModel


TrendRegime = Literal[
    "strong_uptrend",
    "weak_uptrend",
    "range_bound",
    "compression",
    "downtrend",
]

RelativePosition = Literal[
    "cheaper_than_peers",
    "in_line_with_peers",
    "more_expensive_than_peers",
]

GrowthRank = Literal[
    "top_quintile",
    "above_average",
    "average",
    "below_average",
    "bottom_quintile",
]


class PriceFacts(BaseModel):
    last: float
    change_1d_pct: float
    vs_50dma: Literal["above", "below"]
    vs_200dma: Literal["above", "below"]
    trend_regime: TrendRegime


class ValuationFacts(BaseModel):
    relative_position: RelativePosition
    sensitivity: Literal["growth_driven", "margin_driven", "multiple_driven"]


class GrowthFacts(BaseModel):
    revenue_yoy: Optional[float]
    peer_rank: GrowthRank
    trend: Literal["accelerating", "stable", "decelerating"]


class ProfitabilityFacts(BaseModel):
    operating_margin: Optional[float]
    peer_position: Literal["above_median", "in_line", "below_median"]
    trend: Literal["improving", "stable", "deteriorating"]


class LeverageFacts(BaseModel):
    flag: Literal["low", "moderate", "elevated"]
    note: str


class EventFacts(BaseModel):
    next_earnings: str
    importance: Literal["high", "medium", "low"]


class FactsPack(BaseModel):
    price: PriceFacts
    valuation: ValuationFacts
    growth: GrowthFacts
    profitability: ProfitabilityFacts
    leverage: LeverageFacts
    events: EventFacts
    news_flags: List[str]
    sec_flags: List[str]
    data_quality_notes: List[str]
