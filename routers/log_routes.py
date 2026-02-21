import logging
from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


class IngestLogBody(BaseModel):
    level: str = "error"
    message: str
    meta: Optional[dict[str, Any]] = None


@router.post("/log")
async def ingest_log(body: IngestLogBody, request: Request):
    """Accept client-side logs so they appear in Railway backend logs."""
    level = (body.level or "error").lower()
    meta = body.meta or {}
    log_msg = f"[client] {body.message}"
    if meta:
        log_msg += " | " + " ".join(f"{k}={v}" for k, v in meta.items())
    if level == "warn" or level == "warning":
        logger.warning(log_msg)
    elif level == "info":
        logger.info(log_msg)
    elif level == "debug":
        logger.debug(log_msg)
    else:
        logger.error(log_msg)
    return {}
