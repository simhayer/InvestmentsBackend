from typing import TypedDict, Dict, Any, List

class AnalysisOutput(TypedDict, total=False):
    symbol: str
    as_of_utc: str
    pnl_abs: float
    pnl_pct: float
    market_context: Dict[str, Any]
    rating: str                 # "hold" | "sell" | "watch" | "diversify"
    rationale: str
    key_risks: List[str]
    suggestions: List[str]
    data_notes: List[str]
    disclaimer: str
