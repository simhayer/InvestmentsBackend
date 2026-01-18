from __future__ import annotations

from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: str = Field(description="stock_analysis, portfolio, crypto, market_news, education, off_topic")
    symbols: List[str] = Field(default_factory=list, description="Tickers or crypto symbols")
    needs_portfolio: bool = False
    needs_user_profile: bool = True


class ChatState(TypedDict, total=False):
    message: str
    user_id: str
    user_currency: str
    session_id: str

    history: List[Dict[str, str]]
    intent: str
    symbols: List[str]
    needs_portfolio: bool
    needs_user_profile: bool
    fetch_user_profile: bool
    fetch_portfolio_summary: bool
    fetch_holdings: bool
    fetch_fundamentals: bool
    fetch_sec_context: bool
    fetch_sec_business: bool
    fetch_sec_risk: bool
    fetch_sec_mda: bool
    fetch_news: bool

    user_profile: Dict[str, Any]
    portfolio_summary: Dict[str, Any]
    holdings: List[Dict[str, Any]]
    fundamentals: Dict[str, Any]
    fundamentals_gaps: Dict[str, List[str]]
    vector_context: str
    sec_business_context: str
    sec_risk_context: str
    sec_mda_context: str
    news_context: str

    answer: str
    critique: str
    is_valid: bool
    iterations: int
    short_circuit: bool
    fast_path_reason: str
    debug: Dict[str, Any]

    db: Any
    finnhub: Any
