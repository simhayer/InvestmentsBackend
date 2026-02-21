import logging

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from middleware.rate_limit import limiter
from models.user import User
from routers.finnhub_routes import get_finnhub_service
from services.ai.chat.chat_models import ChatRequest
from services.ai.chat.chat_orchestrator import ChatOrchestrator
from services.ai.chat.gemini_stream_client import get_shared_gemini_client
from services.ai.chat.tool_registry import ChatToolRegistry
from services.finnhub.finnhub_service import FinnhubService
from services.supabase_auth import get_current_db_user
from services.tier import require_tier

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/stream")
@limiter.limit("15/minute")
async def stream_chat(
    request: Request,
    payload: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    require_tier(user, db, "chat_messages")
    try:
        tools = ChatToolRegistry(
            finnhub=finnhub,
            db=db,
            user_id=str(user.id),
        )
        orchestrator = ChatOrchestrator(
            gemini_client=get_shared_gemini_client(),
            tools=tools,
            db=db,
            user_id=user.id,
        )
        logger.info("chat_stream_started user_id=%s", user.id)

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
    except Exception as e:
        logger.exception("chat/stream failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})