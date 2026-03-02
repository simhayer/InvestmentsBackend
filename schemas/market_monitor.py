from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MonitorNewsItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    image: Optional[str] = None


class MonitorNewsStream(BaseModel):
    key: str
    label: str
    items: List[MonitorNewsItem] = Field(default_factory=list)


class MonitorMarketPulseItem(BaseModel):
    key: str
    label: str
    symbol: Optional[str] = None
    price: Optional[float] = None
    changeAbs: Optional[float] = None
    changePct: Optional[float] = None
    currency: Optional[str] = None
    sparkline: List[float] = Field(default_factory=list)
    lastUpdated: Optional[str] = None
    error: Optional[str] = None


class MonitorMarketPulseGroup(BaseModel):
    key: str
    label: str
    items: List[MonitorMarketPulseItem] = Field(default_factory=list)


class MonitorAIInsightCard(BaseModel):
    title: str
    summary: str
    signal: Literal["bullish", "bearish", "neutral"]
    time_horizon: str


class MonitorWorldBriefSection(BaseModel):
    headline: str
    cause: str
    impact: str


class MonitorWorldBrief(BaseModel):
    market: str
    sections: List[MonitorWorldBriefSection] = Field(default_factory=list)


class MarketMonitorSections(BaseModel):
    world_brief: MonitorWorldBrief
    ai_insights: List[MonitorAIInsightCard] = Field(default_factory=list)
    market_pulse: List[MonitorMarketPulseGroup] = Field(default_factory=list)
    news_streams: List[MonitorNewsStream] = Field(default_factory=list)


class MarketMonitorMeta(BaseModel):
    sources: List[str] = Field(default_factory=list)
    generated_at: str
    news_categories: List[str] = Field(default_factory=list)


class MarketMonitorPayload(BaseModel):
    as_of: str
    title: str
    subtitle: str
    outlook: Optional[str] = None
    sections: MarketMonitorSections
    meta: MarketMonitorMeta


class WatchlistReference(BaseModel):
    id: int
    name: str
    is_default: bool


class PersonalizedPosition(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    weight: Optional[float] = None
    current_value: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None
    current_price: Optional[float] = None
    currency: Optional[str] = None


class PortfolioInlineInsightsPayload(BaseModel):
    healthBadge: Optional[str] = None
    performanceNote: Optional[str] = None
    riskFlag: Optional[str] = None
    topPerformer: Optional[str] = None
    actionNeeded: Optional[str] = None
    disclaimer: Optional[str] = None


class PortfolioSnapshotPayload(BaseModel):
    as_of: Optional[int] = None
    currency: Optional[str] = None
    price_status: Optional[str] = None
    positions_count: Optional[int] = None
    market_value: Optional[float] = None
    cost_basis: Optional[float] = None
    unrealized_pl: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None
    day_pl: Optional[float] = None
    day_pl_pct: Optional[float] = None
    allocations: Optional[Dict[str, Any]] = None
    connections: Optional[List[Dict[str, Any]]] = None


class PersonalizedFocusNews(BaseModel):
    symbol: str
    items: List[MonitorNewsItem] = Field(default_factory=list)


class PersonalizationPayload(BaseModel):
    scope: Literal["portfolio", "watchlist", "global_fallback"]
    currency: str
    symbols: List[str] = Field(default_factory=list)
    watchlist: Optional[WatchlistReference] = None
    top_positions: List[PersonalizedPosition] = Field(default_factory=list)
    portfolio_snapshot: Optional[PortfolioSnapshotPayload] = None
    inline_insights: Optional[PortfolioInlineInsightsPayload] = None
    insight_cards: List[MonitorAIInsightCard] = Field(default_factory=list)
    focus_news: List[PersonalizedFocusNews] = Field(default_factory=list)
    empty_state: Optional[str] = None


class PersonalizedMarketMonitorPayload(MarketMonitorPayload):
    personalization: PersonalizationPayload


class MarketMonitorEnvelope(BaseModel):
    message: str
    data: MarketMonitorPayload


class PersonalizedMarketMonitorEnvelope(BaseModel):
    message: str
    data: PersonalizedMarketMonitorPayload
