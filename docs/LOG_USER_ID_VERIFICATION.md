# Log user_id verification

This document lists where **user_id** (or an equivalent identifier) appears in logs, and where it does not by design.

---

## Backend – logs that include user_id

All of these run in a request that has an authenticated user (`user` or `current_user` or `_user`), and the log line includes `user_id=%s` (or `user_id=` in the message).

| Location | Log message pattern | Identifier |
|----------|---------------------|------------|
| **main.py** | `request \| METHOD path \| status \| duration user=<sub>` | JWT `sub` (Supabase user id) when Bearer present |
| **holdings_routes** | holding_created, holding_updated, holdings_list, portfolio_history, holding_deleted, edit_holding (warn), delete_holding (warn) | user_id |
| **plaid_routes** | plaid_link_token_created, plaid_token_exchanged, token_exchange_failed, plaid_investments_synced, plaid_update_link_token_created, update_link_token_failed, plaid_institutions_listed, plaid_institutions_failed, plaid_connection_removed, plaid_connection_remove_failed | user_id |
| **portfolio_routes** | portfolio_summary, portfolio summary failed | user_id |
| **v2/analyze_symbol_routes** | symbol_analysis_completed/failed, symbol_inline_insights_completed/failed, symbol_summary_completed/failed, symbol_data_fetched/failed | user_id |
| **v2/analyze_portfolio_routes** | portfolio_analysis_completed/failed, portfolio_inline_insights_completed/failed, portfolio_summary_completed/failed, portfolio_data_fetched/failed | user_id |
| **v2/analyze_crypto_routes** | crypto_analysis_completed/failed, crypto_inline_insights_completed/failed, crypto_data_fetched/failed | user_id |
| **ai_chat_routes** | chat_stream_started, chat_stream_failed | user_id |
| **onboarding_routes** | onboarding_updated, update onboarding profile failed, onboarding_completed | user_id (current_user.id) |
| **user_routes** | currency_updated | user_id (current_user.id) |
| **billing_routes** | checkout_session_created, checkout_session_failed, portal_session_created | user_id; webhook logs use customer_id |

---

## Backend – logs that do not (and should not) include user_id

These either have no authenticated user in the request, or the event is not tied to a single user.

| Location | Reason |
|----------|--------|
| **main.py** | Startup / lifespan (crypto catalog load) – no request |
| **plaid_routes** | Webhook (Plaid → backend): no user in request; logs use webhook_type, webhook_code, item_id. Redirect URI log is config-only. |
| **plaid_routes** | Connection re-auth / expiring: logs use token_entry.id (connection), not user_id (could add user_id from token_entry.user_id if needed) |
| **crypto_routes** | refresh_crypto_catalog – no auth, admin/cron |
| **finnhub_routes** | get_price, get_prices, search_symbols, fetch_quote, fetch_profile – no Depends(get_current_db_user); could add if you add auth later |
| **investment_routes** (Yahoo) | No user dependency on those routes |
| **billing_routes** | stripe_webhook: no user in request; logs use customer_id, plan, status, event type |
| **log_routes** | ingest_log: client sends meta; frontend injects user_id when signed in (see below) |
| **services/supabase_auth** | JWTError, DB retry – we don’t have user id in scope (only token) |
| **services/plaid_sync** | Sync/webhook: has item_id, connection id, or entry; user can be inferred from token_entry.user_id if needed |
| **services/finnhub, yahoo, currency, filings, etc.** | Service-layer logs: no user in scope; some have symbol, request id, or cache key |

---

## Frontend – client logs sent to backend

- **lib/logger.ts** – When `logger.warn()` or `logger.error()` run in the browser, `sendToBackend()` is called. It gets the current session via `supabase.auth.getSession()` and merges **user_id** (Supabase user id) into the `meta` object sent to `POST /api/log`.
- So every client-side error/warn that is sent to the backend includes **user_id** when the user is signed in. If not signed in, `user_id` is omitted.

Frontend **logger.info()** calls do not send to the backend; they only go to the browser console. So they are not in Railway backend logs; no user_id is required there for backend backtracking.

---

## How to search by user in Railway

- **Backend request trail:** `user=<uuid>` or `user_id=<uuid>` in the log line.
- **Client errors (backend):** `[client]` and `user_id=<uuid>` (in the meta string logged by log_routes).
- **Stripe/billing:** `customer_id=<id>` (map to user via your DB if needed).

---

## Summary

- **All authenticated, user-scoped request logs** in the backend include **user_id** (or equivalent).
- **Unauthenticated or system events** (startup, webhooks, cron, third-party) do not include user_id by design; they use item_id, customer_id, or similar where relevant.
- **Client logs** sent to the backend include **user_id** when the user is signed in.
