from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    title: str
    impact: Literal["bullish", "bearish", "mixed", "neutral"]
    explanation: str
    sources: List[str]

    model_config = {"extra": "forbid"}


class FundamentalsSnapshot(BaseModel):
    market_cap: Optional[float] = None
    pe_ttm: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    debt_to_equity: Optional[float] = None
    summary: str

    model_config = {"extra": "forbid"}


class Risk(BaseModel):
    title: str
    severity: Literal["high", "medium", "low"]
    explanation: str
    sources: List[str]

    model_config = {"extra": "forbid"}


class SentimentBlock(BaseModel):
    overall: Literal["positive", "negative", "mixed", "neutral"]
    drivers: List[str]
    sources: List[str]

    model_config = {"extra": "forbid"}


class Scenario(BaseModel):
    thesis: str
    key_assumptions: List[str]
    watch_items: List[str]
    sources: List[str]

    model_config = {"extra": "forbid"}


class Scenarios(BaseModel):
    bull: Scenario
    base: Scenario
    bear: Scenario

    model_config = {"extra": "forbid"}


class ConfidenceBlock(BaseModel):
    score_0_100: int = Field(ge=0, le=100)
    rationale: str

    model_config = {"extra": "forbid"}


class Citation(BaseModel):
    id: str
    title: str
    url: str
    source: str
    published_at: Optional[str] = None

    model_config = {"extra": "forbid"}


class StockReport(BaseModel):
    symbol: str
    as_of: str
    quick_take: str
    what_changed_recently: List[Event]
    fundamentals_snapshot: FundamentalsSnapshot
    catalysts_next_30_90d: List[str]
    risks: List[Risk]
    sentiment: SentimentBlock
    scenarios: Scenarios
    confidence: ConfidenceBlock
    citations: List[Citation]
    data_gaps: List[str]

    model_config = {"extra": "forbid"}
