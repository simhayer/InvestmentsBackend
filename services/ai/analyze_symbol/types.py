from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field

# ----------------------------
# 1) Output schema
# ----------------------------
class AnalysisReport(BaseModel):
    symbol: str
    key_insights: List[str] = Field(description="Critical fundamental highlights")
    current_performance: str = Field(description="Technical and price action analysis")
    stock_overflow_risks: List[str] = Field(description="Red flags and assessment of risks")
    price_outlook: str = Field(description="Deeply reasoned AI outlook balancing bull/bear cases")
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_priced_in: bool = False


# ----------------------------
# 2) Graph state
# ----------------------------
class AgentState(TypedDict, total=False):
    symbol: str
    task_id: str

    raw_data: str
    finnhub_data: Dict[str, Any]
    finnhub_gaps: List[str]

    sec_context: str

    fundamentals: str
    technicals: str
    risks: str

    report: Dict[str, Any]

    critique: str
    is_valid: bool
    iterations: int

    debug: Dict[str, Any]
