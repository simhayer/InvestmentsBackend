from typing import Any, Dict, List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field

Magnitude = Literal["Low", "Medium", "High", "Unknown"]
PricedIn = Literal["Low", "Partial", "High", "Unknown"]


class TextBlock(BaseModel):
    bullets: List[str] = Field(default_factory=list, min_items=1, max_items=6)


class ThesisPoint(BaseModel):
    claim: str
    why_it_matters: str
    what_would_change_my_mind: str = Field(default="Not available")

class Catalyst(BaseModel):
    name: str
    window: str = Field(
        description="YYYY-MM-DD if from earnings_calendar; otherwise use 'Q# YYYY' or timeframe."
    )
    trigger: str
    mechanism: str
    likely_market_reaction: str
    impact_channels: List[str] = Field(default_factory=list)
    probability: float = Field(ge=0.0, le=1.0, default=0.5)
    magnitude: Magnitude = "Unknown"
    priced_in: PricedIn = "Unknown"
    key_watch_items: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    name: Literal["Base", "Bull", "Bear"]
    narrative: str
    key_drivers: List[str] = Field(default_factory=list)
    watch_items: List[str] = Field(default_factory=list)


class DebateItem(BaseModel):
    debate: str
    what_to_watch: List[str] = Field(default_factory=list)


class MarketEdge(BaseModel):
    consensus_view: str
    variant_view: str
    why_it_matters: str


class KeyInsight(BaseModel):
    insight: str
    evidence: Optional[str] = None
    implication: Optional[str] = None


class PeerMetricStat(BaseModel):
    company: Optional[float] = None
    peer_median: Optional[float] = None
    company_percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    peer_count: Optional[int] = None
    higher_is_better: Optional[bool] = None


class PeerComparison(BaseModel):
    peers_used: List[str] = Field(default_factory=list)
    scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    key_stats: Dict[str, PeerMetricStat] = Field(default_factory=dict)


class PricingAssessment(BaseModel):
    market_expectation: str
    variant_outcome: str
    valuation_sensitivity: str


class AnalysisReport(BaseModel):
    symbol: str

    key_insights: List[KeyInsight] = Field(min_items=3, max_items=3)
    unified_thesis: TextBlock

    current_performance: TextBlock
    key_risks: List[str] = Field(min_items=2, max_items=5)
    price_outlook: TextBlock
    what_to_watch_next: List[str] = Field(default_factory=list, min_items=3, max_items=10)

    thesis_points: List[ThesisPoint] = Field(min_items=3, max_items=3)
    upcoming_catalysts: List[Catalyst] = Field(min_items=3, max_items=3)
    scenarios: List[Scenario] = Field(min_items=3, max_items=3)

    key_debates: List[DebateItem] = Field(default_factory=list)

    market_edge: Optional[MarketEdge] = None
    pricing_assessment: PricingAssessment

    recommendation: Literal["Buy", "Hold", "Sell"]
    is_priced_in: bool
    confidence: float = Field(ge=0.0, le=1.0)

    data_quality_notes: List[str] = Field(default_factory=list)

    peer_comparison: Optional[PeerComparison] = None
    peer_comparison_summary: List[str] = Field(default_factory=list)


# ----------------------------
# Graph state
# ----------------------------
class BaseAgentState(TypedDict):
    symbol: str
    iterations: int


class AgentState(BaseAgentState, total=False):
    task_id: str
    raw_data: str

    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]
    market_snapshot: Dict[str, Any]

    peer_benchmark: Dict[str, Any]
    peer_comparison_ready: Dict[str, Any]
    peer_gaps: List[str]

    sec_context: str
    sec_business: List[Dict[str, Any]]
    sec_risks: List[Dict[str, Any]]
    sec_mda: List[Dict[str, Any]]

    earnings_calendar: List[Dict[str, Any]]
    news_items: List[Dict[str, Any]]

    technicals: str

    facts_pack: Dict[str, Any]
    core_analysis: Dict[str, Any]
    report: Dict[str, Any]

    critique: str
    is_valid: bool
    debug: Dict[str, Any]


class CoreAnalysis(BaseModel):
    key_insights: List[KeyInsight] = Field(min_items=3, max_items=3)
    unified_thesis: TextBlock
    thesis_points: List[ThesisPoint] = Field(min_items=3, max_items=3)
    upcoming_catalysts: List[Catalyst] = Field(min_items=3, max_items=3)
    scenarios: List[Scenario] = Field(min_items=3, max_items=3)
    market_edge: Optional[MarketEdge] = None
    pricing_assessment: PricingAssessment
    recommendation: Literal["Buy", "Hold", "Sell"]
    is_priced_in: bool
    confidence: float = Field(ge=0.0, le=1.0)
