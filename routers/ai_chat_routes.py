from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from routers.finnhub_routes import get_finnhub_service
from services.ai.chat.chat_models import ChatRequest
from services.ai.chat.chat_orchestrator import ChatOrchestrator
from services.ai.chat.tool_registry import ChatToolRegistry
from services.finnhub.finnhub_service import FinnhubService
from services.supabase_auth import get_current_db_user

router = APIRouter()


@router.post("/chat/stream")
async def stream_chat(
    request: Request,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    tools = ChatToolRegistry(
        finnhub=finnhub,
        db=db,
        user_id=str(user.id),
    )
    orchestrator = ChatOrchestrator(
        tools=tools,
        db=db,
        user_id=user.id,
    )

    async def event_generator():
        async for event in orchestrator.stream_sse(
            payload,
            is_disconnected=request.is_disconnected,
        ):
            yield event

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )
