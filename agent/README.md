# Decisioning Graph Architecture

This package implements the LangGraph-based Decisioning Graph for the AI stock/portfolio assistant.

## Nodes and Responsibilities
- ingest_request: Deterministic request normalization (intent, tickers, recency, output style, risk flags).
- load_memory: Cache-only memory load (thread summary, recent entities, recent turns, user profile).
- intent_refinement: Small-model intent refinement with strict JSON output (optional override).
- policy_and_budget: Deterministic budgets/caps and allowed data types per intent.
- data_requirements_planner (planner_v1): Small-model planner that selects required/optional DATA TYPES only.
- tool_exec_parallel: Parallel tool execution via ToolExecutor, with caps/timeouts and ToolResult envelopes.
- recency_guard: Flags recency insufficiency when news is missing/stale.
- synthesis (synthesis_v1): Evidence-bound response with guardrails and disclaimers.
- postprocess_and_store: Updates summary/entities and appends turn to CHAT:SESSION.

## Tool Contract
All tools execute through ToolExecutor and return a ToolResult envelope:
- ok: bool
- source: tool name
- as_of: ISO timestamp or null
- latency_ms: int
- warnings: list[str]
- data: tool payload
- error: {type, message, retryable} or null

Tool caps per turn:
- get_holdings: 1
- get_portfolio_summary: 1
- get_news: 1 (max_results=5)
- get_sec_snippets: 2 sections unless explicitly asked
- get_fundamentals: up to 5 tickers
- get_web_search: 1 (max_results=5)

## Memory
Cache-only memory (same TTL as CHAT:SESSION):
- CHAT:SUMMARY:{user_id}:{session_id}
- CHAT:ENTITIES:{user_id}:{session_id}

## Streaming (SSE)
Events:
- plan: safe high-level plan
- tool_status: start/done/timeout with latency/errors
- delta: token or chunk stream
- final: tools_used, trace_id, recency flags

## Observability
Structured logs include trace_id/turn_id, node spans, tool metrics, and eval artifacts.
