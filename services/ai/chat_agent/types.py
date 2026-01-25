from __future__ import annotations

from typing import Any, Dict, List, TypedDict


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

    allowed_tools: List[str]
    tool_caps: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    tool_errors: List[Dict[str, Any]]

    answer: str
    short_circuit: bool
    fast_path_reason: str
    debug: Dict[str, Any]

    db: Any
    finnhub: Any
