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

    user_profile: Dict[str, Any]
    portfolio_summary: Dict[str, Any]
    fundamentals: Dict[str, Any]
    fundamentals_gaps: Dict[str, List[str]]
    vector_context: str

    answer: str
    critique: str
    is_valid: bool
    iterations: int
    debug: Dict[str, Any]

    db: Any
    finnhub: Any
