import json
import os
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from routers.finnhub_routes import get_finnhub_service
from services.finnhub.finnhub_service import FinnhubService
from services.supabase_auth import get_current_db_user
from services.ai.chat_agent.chat_agent_service import (
    run_chat_turn,
    load_chat_history,
    append_chat_history,
)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


def _make_session_id(user_id: Any) -> str:
    return f"sess_{user_id}_{os.urandom(6).hex()}"


def _chunk_text(text: str, size: int = 220) -> List[str]:
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def _sse_pack(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    lines = payload.splitlines() or [""]
    out = f"event: {event}\n" if event else ""
    for line in lines:
        out += f"data: {line}\n"
    out += "\n"
    return out


@router.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    session_id = (req.session_id or "").strip() or _make_session_id(user.id)
    history = load_chat_history(user.id, session_id)

    answer, debug = await run_chat_turn(
        message=message,
        user_id=user.id,
        user_currency=user.currency or "USD",
        session_id=session_id,
        history=history,
        db=db,
        finnhub=finnhub,
    )

    append_chat_history(
        user.id,
        session_id,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "debug": debug,
    }


@router.post("/chat/stream")
async def chat_stream_endpoint(
    req: ChatRequest,
    user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    session_id = (req.session_id or "").strip() or _make_session_id(user.id)
    history = load_chat_history(user.id, session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse_pack("meta", {"session_id": session_id})
        try:
            answer, _debug = await run_chat_turn(
                message=message,
                user_id=user.id,
                user_currency=user.currency or "USD",
                session_id=session_id,
                history=history,
                db=db,
                finnhub=finnhub,
            )
        except Exception as exc:
            yield _sse_pack("error", {"error": str(exc)})
            return

        append_chat_history(
            user.id,
            session_id,
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
        )

        for chunk in _chunk_text(answer):
            yield _sse_pack("delta", chunk)
        yield _sse_pack("done", {"status": "ok"})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
