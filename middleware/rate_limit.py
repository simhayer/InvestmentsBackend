# middleware/rate_limit.py
"""
Rate limiting configuration using slowapi.

Usage in route files:
    from middleware.rate_limit import limiter

    @router.get("/expensive")
    @limiter.limit("5/minute")
    async def my_endpoint(request: Request):
        ...
"""
import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

logger = logging.getLogger(__name__)


def _get_rate_limit_key(request: Request) -> str:
    """
    Identify the caller for rate-limiting.

    Strategy:
      1. If the request carries a valid JWT, use the Supabase user id (sub claim)
         so the limit is per-user regardless of IP.
      2. Otherwise, fall back to client IP.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            from jose import jwt

            # Decode without verification – we only need the 'sub' claim
            # to bucket the rate limit. Auth is enforced separately by the
            # Depends(get_current_db_user) dependency.
            payload = jwt.get_unverified_claims(token)
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass  # fall through to IP-based limiting

    return get_remote_address(request)


# ─── Default limits ────────────────────────────────────────────────
# Env-overridable so you can tune per-environment without redeploying.
DEFAULT_RATE_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")

limiter = Limiter(
    key_func=_get_rate_limit_key,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
    strategy="fixed-window",
)
