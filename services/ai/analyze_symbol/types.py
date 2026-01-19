from typing import Any, Dict, List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field


# ----------------------------
# Pro-grade sub-schemas
# ----------------------------
Magnitude = Literal["Low", "Medium", "High", "Unknown"]
PricedIn = Literal["Low", "Partial", "High", "Unknown"]
CatalystWindow = str  # keep flexible: "Next 30–60 days", "Q2 2026", "Next earnings", etc.

class ThesisPoint(BaseModel):
    """
    One claim + why it matters + what could disprove it.
    This makes the thesis feel like a real analyst note.
    """
    claim: str
    why_it_matters: str
    what_would_change_my_mind: str = Field(default="Not available")

class Catalyst(BaseModel):
    """
    Catalyst = trigger + mechanism + expected impact.
    Ensures dates from earnings_calendar are prioritized.
    """
    name: str = Field(description="Short name of the event, e.g., 'Q1 Earnings Release'")
    
    # Updated window field to force date formatting
    window: str = Field(
        description=(
            "The specific date or timeframe. "
            "STRICT RULE: If a date is provided in the earnings_calendar, "
            "you MUST use the YYYY-MM-DD format here. Otherwise, use 'Q# YYYY'."
        ),
        examples=["2026-01-29", "2026-04-29"]
    )
    
    trigger: str = Field(description="Specific event or data point release.")
    mechanism: str = Field(description="How the event affects valuation (e.g., 'beat leads to multiple expansion').")
    likely_market_reaction: str = Field(description="Expected direction + reasoning (no price targets).")
    
    impact_channels: List[str] = Field(
        default_factory=list,
        description="Pick from: revenue, margins, guidance, multiple, risk_premium, liquidity, sentiment, regulation"
    )
    
    probability: float = Field(ge=0.0, le=1.0, default=0.5)
    magnitude: Magnitude = "Unknown"
    priced_in: PricedIn = "Unknown"
    key_watch_items: List[str] = Field(default_factory=list, description="Concrete metrics to watch.")

class Scenario(BaseModel):
    """
    Scenario framing without numeric price targets.
    """
    name: Literal["Base", "Bull", "Bear"]
    narrative: str = Field(description="What happens and why.")
    key_drivers: List[str] = Field(default_factory=list)
    watch_items: List[str] = Field(default_factory=list)


class DebateItem(BaseModel):
    """
    Real analyst notes include the 2–4 things the market disagrees about.
    """
    debate: str = Field(description="e.g., 'Margin durability vs competitive pressure'")
    what_to_watch: List[str] = Field(default_factory=list)

class MarketEdge(BaseModel):
    consensus_view: str = Field(description="What the market currently assumes.")
    variant_view: str = Field(description="Why that assumption may be wrong or fragile.")
    why_it_matters: str = Field(description="Implications if the variant view is correct.")

class KeyInsight(BaseModel):
    insight: str
    evidence: Optional[str]
    implication: Optional[str]

# ----------------------------
# Output schema (upgraded)
# ----------------------------
class AnalysisReport(BaseModel):
    # Keep existing fields
    symbol: str
    key_insights: List[KeyInsight] = Field(description="Critical fundamental highlights")
    current_performance: str = Field(description="Technical and price action analysis")
    stock_overflow_risks: List[str] = Field(description="Red flags and assessment of risks")
    price_outlook: str = Field(description="Deeply reasoned AI outlook balancing bull/bear cases")
    recommendation: Literal["Buy", "Hold", "Sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    is_priced_in: bool = False

    # New pro-grade fields
    unified_thesis: str = Field(description="A single coherent view: what drives the stock and why now.")

    thesis_points: List[ThesisPoint] = Field(
        default_factory=list,
        description="3–6 structured thesis claims with falsifiers."
    )

    upcoming_catalysts: List[Catalyst] = Field(
        default_factory=list,
        description="3–8 catalysts with trigger → mechanism → impact."
    )

    scenarios: List[Scenario] = Field(
        default_factory=list,
        description="Base/Bull/Bear scenario narratives (no price targets)."
    )

    market_expectations: List[str] = Field(
        default_factory=list,
        description="What the current setup/valuation seems to imply the market expects."
    )
    key_debates: List[DebateItem] = Field(
        default_factory=list,
        description="2–4 disagreements investors have + what would settle them."
    )

    what_to_watch_next: List[str] = Field(
        default_factory=list,
        description="5–10 concrete watch items (metrics, events, statements)."
    )

    data_quality_notes: List[str] = Field(
        default_factory=list,
        description="Call out missing fundamentals/weak news/empty SEC context."
    )

    market_edge: Optional[MarketEdge] = Field(
        default=None,
        description="Where is the market likely wrong? Consensus vs variant view."
    )

    pricing_assessment: Optional[Dict[str, str]] = Field(
        default=None,
        description="What is priced in vs not priced in."
    )


# ----------------------------
# Graph state (upgraded)
# ----------------------------
class BaseAgentState(TypedDict):
    symbol: str
    iterations: int

class AgentState(BaseAgentState, total=False):
    task_id: str
    raw_data: str
    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]

    sec_context: str
    sec_business: List[Dict[str, Any]]
    sec_risks: List[Dict[str, Any]]
    sec_mda: List[Dict[str, Any]]

    earnings_calendar: List[Dict[str, Any]]
    news_items: List[Dict[str, Any]]  # <-- add this if not already present

    fundamentals: str
    technicals: str  # now deterministic text
    risks: str

    report: Dict[str, Any]
    critique: str
    is_valid: bool
    debug: Dict[str, Any]
