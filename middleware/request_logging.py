"""
Request logging middleware. Logs method, path, status, duration only.
Never logs headers, body, or query params (may contain tokens or PII).
"""
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request: method, path (no query), status_code, duration_ms."""

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        # Log path only; do not log query string (may contain tokens or PII)
        path = request.scope.get("path", "")
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        if status >= 500:
            logger.error(
                "request_finished method=%s path=%s status=%s duration_ms=%.1f",
                method, path, status, duration_ms,
            )
        elif status >= 400:
            logger.warning(
                "request_finished method=%s path=%s status=%s duration_ms=%.1f",
                method, path, status, duration_ms,
            )
        else:
            logger.info(
                "request_finished method=%s path=%s status=%s duration_ms=%.1f",
                method, path, status, duration_ms,
            )
        return response
