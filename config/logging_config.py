"""
Central logging configuration for the backend.

- Railway-friendly: JSON logs when LOG_JSON=1 or RAILWAY_ENVIRONMENT is set.
- LOG_LEVEL from env (default INFO).
- Never log PII: no emails, tokens, user identifiers in messages.
  Use this module's logger or getLogger(__name__) and avoid passing
  customer data into log messages.
"""
import json
import logging
import os
import sys
from datetime import timezone
from typing import Any


def _json_serial(obj: Any):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for Railway/log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            payload["exception"] = self.formatException(record.exc_info)
        # Optional: add extra fields if present (e.g. request_id)
        if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
            for k, v in getattr(record, "extra").items():
                if k not in payload and v is not None:
                    payload[k] = v
        return json.dumps(payload, default=_json_serial)


def configure_logging() -> None:
    """Configure root logger: level from LOG_LEVEL, JSON format in production."""
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    use_json = (
        os.getenv("LOG_JSON", "").lower() in ("1", "true", "yes")
        or bool(os.getenv("RAILWAY_ENVIRONMENT"))
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers when reloading
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(handler)

    # Reduce noise from third-party libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
