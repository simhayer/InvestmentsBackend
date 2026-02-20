from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(max_length=8000)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        return (v or "").strip()


# ── Page Context (from frontend) ────────────────────────────────────────

class PageContext(BaseModel):
    page_type: str = Field(max_length=32)
    route: str = Field(max_length=256)
    symbol: Optional[str] = Field(default=None, max_length=20)
    summary: Optional[str] = Field(default=None, max_length=3000)
    data_snapshot: Optional[Dict[str, Any]] = None


class ChatContext(BaseModel):
    portfolio_summary: Optional[str] = Field(default=None, max_length=8000)
    risk_profile: Optional[str] = Field(default=None, max_length=200)
    preferred_currency: Optional[str] = Field(default=None, max_length=8)
    page: Optional[PageContext] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(max_length=40)
    context: Optional[ChatContext] = None
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    allow_web_search: Optional[bool] = None

    @model_validator(mode="after")
    def _drop_empty_messages(self) -> "ChatRequest":
        self.messages = [m for m in self.messages if m.content]
        if not self.messages:
            raise ValueError("At least one message with non-empty content is required")
        return self


SSEEventType = Literal[
    "meta",
    "tool_call",
    "tool_result",
    "token",
    "done",
    "error",
    "heartbeat",
    "thinking",
    "page_ack",
    "plan",
    "citation",
]


class SSEEvent(BaseModel):
    event: SSEEventType
    data: Dict[str, Any]


def format_sse(event: str, data: Dict[str, Any]) -> str:
    import json

    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"
