"""
Type definitions for stock analysis service.

Enhanced with:
- Better validation constraints
- More comprehensive documentation
- Improved type safety
- Helper methods for common operations
"""

from typing import Any, Dict, List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

# ============================================================================
# LITERALS & CONSTANTS
# ============================================================================

Magnitude = Literal["Low", "Medium", "High", "Unknown"]
PricedIn = Literal["Low", "Partial", "High", "Unknown"]
NewsRelevance = Literal["high", "medium", "low"]
Recommendation = Literal["Buy", "Hold", "Sell"]
Direction = Literal["risk", "opportunity"]
ScenarioName = Literal["Base", "Bull", "Bear"]


# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

class TextBlock(BaseModel):
    """
    Text content represented as bullet points.
    Used for summaries, outlooks, and narratives.
    """
    bullets: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=6,
        description="List of bullet points (1-6 items)"
    )

    @field_validator('bullets')
    @classmethod
    def validate_bullets_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure no empty bullets."""
        non_empty = [b.strip() for b in v if b.strip()]
        if len(non_empty) < 1:
            raise ValueError("TextBlock must have at least 1 non-empty bullet")
        return non_empty


# ============================================================================
# NEWS BRIEF COMPONENTS
# ============================================================================

class NewsBullet(BaseModel):
    """
    Individual news item with evidence and source.
    """
    bullet: str = Field(
        ...,
        min_length=8,
        max_length=180,
        description="Brief description of the news item"
    )
    evidence: str = Field(
        ...,
        min_length=8,
        max_length=220,
        description="Direct quote or paraphrase from source"
    )
    source: Optional[str] = Field(
        default=None,
        max_length=80,
        description="News source name"
    )
    published_at: Optional[str] = Field(
        default=None,
        description="Publication timestamp (ISO format preferred)"
    )
    url: Optional[str] = Field(
        default=None,
        description="Source URL"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the interpretation (0.0-1.0)"
    )


class CatalystCandidate(BaseModel):
    """
    Potential upcoming catalyst identified from news.
    """
    name: str = Field(
        ...,
        min_length=4,
        max_length=80,
        description="Short catalyst name"
    )
    window: str = Field(
        ...,
        min_length=2,
        max_length=30,
        description="Expected timing: 'YYYY-MM-DD' or 'Q# YYYY'"
    )
    trigger: str = Field(
        ...,
        min_length=6,
        max_length=140,
        description="What event triggers the catalyst"
    )
    mechanism: str = Field(
        ...,
        min_length=10,
        max_length=220,
        description="How it affects the stock"
    )
    evidence: str = Field(
        ...,
        min_length=8,
        max_length=220,
        description="Supporting evidence from news"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence this catalyst will materialize"
    )

    @field_validator('window')
    @classmethod
    def validate_window_format(cls, v: str) -> str:
        """Validate window is either YYYY-MM-DD or Q# YYYY format."""
        v = v.strip()
        # Check YYYY-MM-DD format
        if len(v) == 10 and v[4] == '-' and v[7] == '-':
            try:
                datetime.strptime(v, '%Y-%m-%d')
                return v
            except ValueError:
                pass
        # Check Q# YYYY format (e.g., "Q1 2024")
        if v.startswith('Q') and len(v) >= 7:
            return v
        # Allow other timeframes like "Early 2024", "H1 2024"
        return v


class NewsBrief(BaseModel):
    """
    Daily news brief with key changes, themes, and catalysts.
    
    This is the output of the news_brief_node and drives
    whether raw news is included in downstream analysis.
    """
    what_changed_today: List[NewsBullet] = Field(
        min_length=2,
        max_length=4,
        description="Key developments from today's news"
    )
    key_themes: List[str] = Field(
        min_length=2,
        max_length=4,
        description="Overarching themes from recent news"
    )
    catalyst_candidates: List[CatalystCandidate] = Field(
        min_length=2,
        max_length=4,
        description="Potential upcoming catalysts"
    )
    risk_signals: List[NewsBullet] = Field(
        default_factory=list,
        max_length=4,
        description="Warning signs from news (optional)"
    )
    news_relevance: NewsRelevance = Field(
        default="medium",
        description="Overall news quality: high/medium/low"
    )


# ============================================================================
# MATERIALITY RESEARCH
# ============================================================================

class MaterialDriver(BaseModel):
    """
    Material risk or opportunity with evidence and watch items.
    """
    label: str = Field(
        ...,
        min_length=3,
        max_length=80,
        description="Short label for the driver"
    )
    direction: Direction = Field(
        ...,
        description="Whether this is a risk or opportunity"
    )
    mechanism: str = Field(
        ...,
        min_length=10,
        max_length=240,
        description="How this impacts revenue/margins/valuation"
    )
    evidence: List[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="1-2 evidence snippets from sources"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in materiality (0.0-1.0)"
    )
    watch_items: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Key metrics/events to monitor"
    )

    @field_validator('evidence')
    @classmethod
    def validate_evidence_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure evidence items are non-empty."""
        non_empty = [e.strip() for e in v if e.strip()]
        if len(non_empty) < 1:
            raise ValueError("MaterialDriver must have at least 1 evidence item")
        return non_empty


class RiskResearchOutput(BaseModel):
    """
    Output of risk_research_node: material risks, opportunities, and watch list.
    """
    risks: List[MaterialDriver] = Field(
        default_factory=list,
        max_length=5,
        description="Material downside risks"
    )
    opportunities: List[MaterialDriver] = Field(
        default_factory=list,
        max_length=5,
        description="Material upside opportunities"
    )
    watch_list: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Key items to monitor"
    )

    @model_validator(mode='after')
    def validate_has_content(self) -> 'RiskResearchOutput':
        """Ensure at least some risks or opportunities identified."""
        if not self.risks and not self.opportunities and not self.watch_list:
            raise ValueError(
                "RiskResearchOutput must have at least one risk, "
                "opportunity, or watch item"
            )
        return self


# ============================================================================
# CORE ANALYSIS COMPONENTS
# ============================================================================

class KeyInsight(BaseModel):
    """
    Key investment insight with evidence and implication.
    """
    insight: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="The core observation"
    )
    evidence: Optional[str] = Field(
        default=None,
        max_length=400,
        description="Supporting evidence with specific metrics"
    )
    implication: Optional[str] = Field(
        default=None,
        max_length=300,
        description="What this means for the investment"
    )

    @field_validator('evidence')
    @classmethod
    def validate_evidence_has_numbers(cls, v: Optional[str]) -> Optional[str]:
        """Evidence should ideally contain numbers."""
        if v and not any(c.isdigit() for c in v):
            # This is a warning, not a hard error
            pass
        return v


class ThesisPoint(BaseModel):
    """
    Investment thesis point with claim and supporting logic.
    """
    claim: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="The thesis claim"
    )
    why_it_matters: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="Why this matters for the investment"
    )
    what_would_change_my_mind: str = Field(
        default="Not available",
        max_length=300,
        description="What evidence would invalidate this thesis"
    )


class Catalyst(BaseModel):
    """
    Upcoming catalyst with detailed analysis.
    """
    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Catalyst name"
    )
    window: str = Field(
        ...,
        description="Timing: 'YYYY-MM-DD' from earnings or 'Q# YYYY'"
    )
    trigger: str = Field(
        ...,
        min_length=8,
        max_length=200,
        description="What triggers this catalyst"
    )
    mechanism: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="How it affects the stock"
    )
    likely_market_reaction: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Expected market response"
    )
    impact_channels: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="How impact flows through (revenue, margins, etc.)"
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Probability of occurrence"
    )
    magnitude: Magnitude = Field(
        default="Unknown",
        description="Expected impact magnitude"
    )
    priced_in: PricedIn = Field(
        default="Unknown",
        description="How much is already priced in"
    )
    key_watch_items: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Signals to monitor"
    )


class Scenario(BaseModel):
    """
    Investment scenario (Base/Bull/Bear).
    """
    name: ScenarioName = Field(
        ...,
        description="Scenario name: Base, Bull, or Bear"
    )
    narrative: str = Field(
        ...,
        min_length=20,
        max_length=500,
        description="Scenario description"
    )
    key_drivers: List[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=5,
        description="Key factors driving this scenario"
    )
    watch_items: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="What to watch to track this scenario"
    )


class MarketEdge(BaseModel):
    """
    Non-consensus view and why it matters.
    Required when confidence >= 0.6.
    """
    consensus_view: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="What the market currently believes"
    )
    variant_view: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="How our view differs"
    )
    why_it_matters: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Why this difference creates opportunity"
    )


class PricingAssessment(BaseModel):
    """
    Assessment of what's priced in and valuation sensitivity.
    """
    market_expectation: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="What the market expects"
    )
    variant_outcome: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Potential variant outcome"
    )
    valuation_sensitivity: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="How valuation responds to key variables"
    )


class DebateItem(BaseModel):
    """
    Key debate or uncertainty with watch items.
    """
    debate: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="The key debate or question"
    )
    what_to_watch: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Signals that will resolve the debate"
    )


# ============================================================================
# PEER COMPARISON
# ============================================================================

class PeerMetricStat(BaseModel):
    """
    Single metric comparison to peer group.
    """
    company: Optional[float] = Field(
        default=None,
        description="Company's value for this metric"
    )
    peer_median: Optional[float] = Field(
        default=None,
        description="Peer group median"
    )
    company_percentile: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Company's percentile rank (0-100)"
    )
    peer_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of peers in comparison"
    )
    higher_is_better: Optional[bool] = Field(
        default=None,
        description="Whether higher values are better"
    )


class PeerComparison(BaseModel):
    """
    Comprehensive peer comparison data.
    """
    peers_used: List[str] = Field(
        default_factory=list,
        description="List of peer ticker symbols"
    )
    scores: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Peer scores/rankings"
    )
    key_stats: Dict[str, PeerMetricStat] = Field(
        default_factory=dict,
        description="Key metric comparisons"
    )


# ============================================================================
# CORE ANALYSIS OUTPUT
# ============================================================================

class CoreAnalysis(BaseModel):
    """
    Core investment analysis output from analyst_core_node.
    
    This contains the main LLM-generated insights and thesis.
    """
    key_insights: List[KeyInsight] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 key insights with evidence"
    )
    unified_thesis: TextBlock = Field(
        ...,
        description="Overall investment thesis"
    )
    thesis_points: List[ThesisPoint] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 detailed thesis points"
    )
    upcoming_catalysts: List[Catalyst] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 upcoming catalysts"
    )
    scenarios: List[Scenario] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 scenarios: Base, Bull, Bear"
    )
    market_edge: Optional[MarketEdge] = Field(
        default=None,
        description="Non-consensus view (required if confidence >= 0.6)"
    )
    pricing_assessment: PricingAssessment = Field(
        ...,
        description="What's priced in and valuation sensitivity"
    )
    recommendation: Recommendation = Field(
        ...,
        description="Buy/Hold/Sell recommendation"
    )
    is_priced_in: bool = Field(
        ...,
        description="Whether key thesis is already priced in"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in analysis (0.0-1.0)"
    )

    @model_validator(mode='after')
    def validate_scenarios(self) -> 'CoreAnalysis':
        """Ensure exactly Base, Bull, Bear scenarios."""
        scenario_names = sorted([s.name for s in self.scenarios])
        expected = ["Base", "Bear", "Bull"]
        if scenario_names != expected:
            raise ValueError(
                f"Scenarios must be exactly Base, Bull, Bear. Got: {scenario_names}"
            )
        return self

    @model_validator(mode='after')
    def validate_market_edge_when_high_confidence(self) -> 'CoreAnalysis':
        """Market edge required when confidence >= 0.6."""
        if self.confidence >= 0.6 and self.market_edge is None:
            raise ValueError(
                "market_edge is required when confidence >= 0.6"
            )
        return self


# ============================================================================
# FINAL ANALYSIS REPORT
# ============================================================================

class AnalysisReport(BaseModel):
    """
    Complete stock analysis report for end user.
    
    This is the final output assembled from all graph nodes.
    """
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )

    # Core insights
    key_insights: List[KeyInsight] = Field(
        min_length=3,
        max_length=3,
        description="3 key investment insights"
    )
    unified_thesis: TextBlock = Field(
        ...,
        description="Overall investment thesis"
    )

    # Deterministic sections
    current_performance: TextBlock = Field(
        ...,
        description="Current performance summary"
    )
    key_risks: List[str] = Field(
        min_length=2,
        max_length=5,
        description="Material risks"
    )
    price_outlook: TextBlock = Field(
        ...,
        description="Price trajectory and outlook"
    )
    what_to_watch_next: List[str] = Field(
        default_factory=list,
        min_length=3,
        max_length=10,
        description="Key items to monitor"
    )

    # Detailed analysis
    thesis_points: List[ThesisPoint] = Field(
        min_length=3,
        max_length=3,
        description="3 detailed thesis points"
    )
    upcoming_catalysts: List[Catalyst] = Field(
        min_length=3,
        max_length=3,
        description="3 upcoming catalysts"
    )
    scenarios: List[Scenario] = Field(
        min_length=3,
        max_length=3,
        description="Base/Bull/Bear scenarios"
    )

    # Optional sections
    key_debates: List[DebateItem] = Field(
        default_factory=list,
        description="Key open questions"
    )
    market_edge: Optional[MarketEdge] = Field(
        default=None,
        description="Non-consensus view"
    )
    pricing_assessment: PricingAssessment = Field(
        ...,
        description="Pricing and valuation sensitivity"
    )

    # Final verdict
    recommendation: Recommendation = Field(
        ...,
        description="Buy/Hold/Sell"
    )
    is_priced_in: bool = Field(
        ...,
        description="Whether thesis is priced in"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence (0.0-1.0)"
    )

    # Quality metadata
    data_quality_notes: List[str] = Field(
        default_factory=list,
        description="Data quality/completeness notes"
    )

    # Peer comparison
    peer_comparison: Optional[PeerComparison] = Field(
        default=None,
        description="Peer comparison data"
    )
    peer_comparison_summary: List[str] = Field(
        default_factory=list,
        description="Peer comparison summary bullets"
    )


# ============================================================================
# GRAPH STATE
# ============================================================================

class BaseAgentState(TypedDict):
    """Base required fields for agent state."""
    symbol: str
    iterations: int


class AgentState(BaseAgentState, total=False):
    """
    Complete agent state for LangGraph workflow.
    
    This tracks all intermediate data as it flows through nodes.
    """
    # Task tracking
    task_id: str

    # Raw data
    raw_data: str

    # Fundamentals
    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]
    market_snapshot: Dict[str, Any]

    # Peer analysis
    peer_benchmark: Dict[str, Any]
    peer_comparison_ready: Dict[str, Any]
    peer_gaps: List[str]

    # SEC filings
    sec_context: str
    sec_business: List[Dict[str, Any]]
    sec_risks: List[Dict[str, Any]]
    sec_mda: List[Dict[str, Any]]

    # Events & news
    earnings_calendar: List[Dict[str, Any]]
    news_items: List[Dict[str, Any]]
    news_brief: Dict[str, Any]

    # Technical analysis
    technicals: str

    # Processed data
    facts_pack: Dict[str, Any]
    risk_research: Dict[str, Any]
    core_analysis: Dict[str, Any]

    # Final output
    report: Dict[str, Any]

    # Quality tracking
    critique: str
    is_valid: bool
    quality_warnings: List[str]
    data_completeness: Dict[str, bool]
    debug: Dict[str, Any]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Literals
    "Magnitude",
    "PricedIn",
    "NewsRelevance",
    "Recommendation",
    "Direction",
    "ScenarioName",
    
    # Basic types
    "TextBlock",
    
    # News brief
    "NewsBullet",
    "CatalystCandidate",
    "NewsBrief",
    
    # Materiality
    "MaterialDriver",
    "RiskResearchOutput",
    
    # Core analysis
    "KeyInsight",
    "ThesisPoint",
    "Catalyst",
    "Scenario",
    "MarketEdge",
    "PricingAssessment",
    "DebateItem",
    
    # Peer comparison
    "PeerMetricStat",
    "PeerComparison",
    
    # Outputs
    "CoreAnalysis",
    "AnalysisReport",
    
    # State
    "AgentState",
    "BaseAgentState",
]