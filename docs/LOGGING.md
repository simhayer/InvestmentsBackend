# Backend Logging (Railway / Production)

## Overview

Logging is configured for [Railway](https://railway.com/) and other environments. Logs go to stdout; never log customer personal data (PII).

## Configuration

- **LOG_LEVEL**: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`).
- **LOG_JSON**: set to `1` (or `true`) for JSON lines (recommended on Railway).
- **RAILWAY_ENVIRONMENT**: when set, JSON logging is enabled automatically.

Configured in `config/logging_config.py` and applied at app startup in `main.py`.

## What We Log

- **Request logging** (middleware): `method`, `path` (no query string), `status_code`, `duration_ms`. No headers or body.
- **Startup**: crypto catalog load, DB pool config.
- **Auth**: JWT validation failures (no token or error details), new user creation (no email/id).
- **Errors**: exception type and message where safe; no request bodies or tokens.
- **Business events**: subscription events (no customer id), cache hits/misses (no user id), filings/symbols, external API failures (symbol/url only where relevant).

## What We Never Log (PII / Sensitive)

- **Never**: email, name, phone, address, IP (beyond what Railway might add), any token (JWT, Plaid, Stripe, API keys).
- **Never**: request/response bodies, query strings that might contain tokens.
- **Never**: internal user id, Stripe customer id, Supabase user id, account names, institution names, portfolio summaries or holding details.
- **Never**: full tracebacks in handlers that process tokens or PII (e.g. Plaid token exchange).

Symbols, tickers, and public URLs (e.g. filing URLs) are considered non-PII and may be logged where useful for debugging.

## Adding New Logs

1. Use `logging.getLogger(__name__)` and standard levels: `debug`, `info`, `warning`, `error`, `exception` (only where no PII in stack).
2. Prefer structured keys: `logger.info("event_name key1=%s key2=%s", v1, v2)`.
3. Do not pass user input, tokens, or identifiers into log messages; use opaque descriptions (e.g. "Token exchange failed error_type=ValueError") or redact.

## Request Flow

1. `config.logging_config.configure_logging()` runs first in `main.py`.
2. `RequestLoggingMiddleware` runs for each request and logs method, path, status, duration.
3. Routers and services use their own loggers; all inherit root config (level + JSON/human format).
