from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict

from pydantic import BaseModel, Field, ConfigDict

IntentType = Literal[
    "portfolio_q",
    "single_stock_q",
    "news_q",
    "sec_q",
    "education_q",
    "meta_q",
]

DataType = Literal[
    "portfolio_summary",
    "holdings",
    "fundamentals",
    "news",
    "sec_snippets",
    "web_search",
]

SecSection = Literal["risk", "mda", "business", "general"]
OutputStyle = Literal["short", "long"]
RiskFlag = Literal["panic_sell", "day_trading", "options"]


class RequestContext(BaseModel):
    intent: IntentType = "education_q"
    tickers: List[str] = Field(default_factory=list)
    needs_portfolio: bool = False
    needs_recency: bool = False
    requested_sections: Optional[List[SecSection]] = None
    output_style: OutputStyle = "short"
    risk_flags: Set[RiskFlag] = Field(default_factory=set)
    timeframe: Optional[str] = None
    user_constraints: List[str] = Field(default_factory=list)


class MemorySnapshot(BaseModel):
    thread_summary: str = ""
    recent_entities: List[str] = Field(default_factory=list)
    recent_turns: List[Dict[str, str]] = Field(default_factory=list)
    user_profile: Dict[str, Any] = Field(default_factory=dict)


class DataRequirementsPlan(BaseModel):
    required_data: List[DataType] = Field(default_factory=list)
    optional_data: List[DataType] = Field(default_factory=list)
    sec_sections: List[SecSection] = Field(default_factory=list)
    notes: str = ""


class ToolError(BaseModel):
    type: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    ok: bool
    source: str
    as_of: Optional[str] = None
    latency_ms: int = 0
    warnings: List[str] = Field(default_factory=list)
    data: Any = None
    error: Optional[ToolError] = None


class ToolBudget(BaseModel):
    max_calls: int = 1
    timeout_s: float = 6.0
    max_items: Optional[int] = None
    max_results: Optional[int] = None
    max_symbols: Optional[int] = None
    max_sections: Optional[int] = None
    max_snippets: Optional[int] = None


class BudgetConfig(BaseModel):
    global_timeout_s: float = 12.0
    tool_budgets: Dict[str, ToolBudget] = Field(default_factory=dict)
    allowed_data_types: List[DataType] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    answer: str


class IntentRefinementOutput(BaseModel):
    intent: Optional[IntentType] = None
    needs_portfolio: Optional[bool] = None
    needs_recency: Optional[bool] = None
    requested_sections: List[SecSection] = Field(default_factory=list)
    output_style: Optional[OutputStyle] = None
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    notes: str = ""


class GraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    user_id: Any
    user_currency: str
    session_id: str
    trace_id: str
    turn_id: str
    request_context: Optional[RequestContext] = None
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    data_requirements: DataRequirementsPlan = Field(default_factory=DataRequirementsPlan)
    tool_results: List[ToolResult] = Field(default_factory=list)
    tool_statuses: List[Dict[str, Any]] = Field(default_factory=list)
    recency_insufficient: bool = False
    answer: str = ""
    db: Any = None
    finnhub: Any = None
    debug: Dict[str, Any] = Field(default_factory=dict)


class GraphStateDict(TypedDict, total=False):
    message: str
    user_id: Any
    user_currency: str
    session_id: str
    trace_id: str
    turn_id: str
    request_context: RequestContext
    memory: MemorySnapshot
    budgets: BudgetConfig
    data_requirements: DataRequirementsPlan
    tool_results: List[ToolResult]
    tool_statuses: List[Dict[str, Any]]
    recency_insufficient: bool
    answer: str
    db: Any
    finnhub: Any
    debug: Dict[str, Any]


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
