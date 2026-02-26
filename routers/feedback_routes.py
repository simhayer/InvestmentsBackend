import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from jose import JWTError, jwt

from middleware.rate_limit import limiter
from schemas.feedback import FeedbackCreate

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])

SUPABASE_PROJECT_URL = os.getenv("SUPABASE_PROJECT_URL", "").rstrip("/")
SUPABASE_JWT_AUD = os.getenv("SUPABASE_JWT_AUD", "authenticated")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def _extract_user_id(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    if not token or not SUPABASE_JWT_SECRET or not SUPABASE_PROJECT_URL:
        return None

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience=SUPABASE_JWT_AUD,
            issuer=f"{SUPABASE_PROJECT_URL}/auth/v1",
        )
    except JWTError:
        return None

    subject = payload.get("sub")
    if not subject:
        return None
    return str(subject)


def _safe_response_detail(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except json.JSONDecodeError:
        return resp.text[:500]


@router.post("/submit")
@limiter.limit("20/minute")
async def submit_feedback(body: FeedbackCreate, request: Request):
    if not SUPABASE_PROJECT_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.error(
            "feedback_submit_missing_supabase_config project_url_set=%s service_key_set=%s",
            bool(SUPABASE_PROJECT_URL),
            bool(SUPABASE_SERVICE_ROLE_KEY),
        )
        raise HTTPException(status_code=500, detail="Feedback service is not configured")

    user_id = _extract_user_id(request)
    payload = {
        "message": body.message,
        "category": body.category,
        "email": body.email,
        "page_url": body.page_url,
        "user_id": user_id,
    }

    url = f"{SUPABASE_PROJECT_URL}/rest/v1/user_feedback"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.exception("feedback_submit_supabase_unreachable error=%s", exc)
        raise HTTPException(status_code=502, detail="Failed to store feedback")

    if resp.status_code >= 400:
        logger.error(
            "feedback_submit_supabase_error status=%s body=%s",
            resp.status_code,
            _safe_response_detail(resp),
        )
        raise HTTPException(status_code=502, detail="Failed to store feedback")

    return {"ok": True}
