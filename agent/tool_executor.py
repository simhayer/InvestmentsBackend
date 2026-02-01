from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from services.ai.chat_agent.tools import TOOL_REGISTRY, ToolContext
from services.holding_service import get_holdings_with_live_prices

from .state_models import ToolBudget, ToolError, ToolResult, now_iso

logger = logging.getLogger("decisioning_graph.tools")


@dataclass(frozen=True)
class ToolCallSpec:
    name: str
    arguments: Dict[str, Any]
    data_type: str


class ToolExecutor:
    def __init__(
        self,
        tool_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._registry = tool_registry or TOOL_REGISTRY

    async def execute(
        self,
        calls: List[ToolCallSpec],
        ctx: ToolContext,
        budgets: Dict[str, ToolBudget],
        global_timeout_s: float,
    ) -> Tuple[List[ToolResult], List[Dict[str, Any]]]:
        results: List[ToolResult] = []
        statuses: List[Dict[str, Any]] = []
        if not calls:
            return results, statuses

        counts: Dict[str, int] = {}
        prepared_calls: List[ToolCallSpec] = []
        for call in calls:
            counts[call.name] = counts.get(call.name, 0) + 1
            budget = budgets.get(call.name, ToolBudget())
            if counts[call.name] > budget.max_calls:
                results.append(
                    ToolResult(
                        ok=False,
                        source=call.name,
                        as_of=None,
                        latency_ms=0,
                        warnings=[],
                        data=None,
                        error=ToolError(
                            type="cap_exceeded",
                            message=f"max_calls exceeded for {call.name}",
                            retryable=False,
                        ),
                    )
                )
                statuses.append(
                    {
                        "name": call.name,
                        "status": "done",
                        "latency_ms": 0,
                        "error_type": "cap_exceeded",
                    }
                )
                continue
            prepared_calls.append(call)

        holdings_snapshot = await self._maybe_holdings_snapshot(prepared_calls, ctx)
        tool_ctx = ToolContext(
            db=ctx.db,
            finnhub=ctx.finnhub,
            user_id=ctx.user_id,
            user_currency=ctx.user_currency,
            message=ctx.message,
            symbols=ctx.symbols,
            history=ctx.history,
            holdings_snapshot=holdings_snapshot,
        )

        tasks: Dict[asyncio.Task[ToolResult], Dict[str, Any]] = {}

        for idx, call in enumerate(prepared_calls):
            call_id = f"{call.name}_{idx}"
            start_time = time.perf_counter()
            statuses.append({"name": call.name, "status": "start", "call_id": call_id})
            task = asyncio.create_task(self._invoke_tool(call, tool_ctx, budgets))
            tasks[task] = {"call": call, "call_id": call_id, "start": start_time}

        if not tasks:
            return results, statuses

        done, pending = await asyncio.wait(tasks.keys(), timeout=global_timeout_s)
        for task in done:
            meta = tasks[task]
            call = meta["call"]
            call_id = meta["call_id"]
            start_time = meta["start"]
            try:
                res = task.result()
            except Exception as exc:
                res = ToolResult(
                    ok=False,
                    source=call.name,
                    as_of=None,
                    latency_ms=self._elapsed_ms(start_time),
                    warnings=[],
                    data=None,
                    error=ToolError(
                        type="tool_error",
                        message=str(exc),
                        retryable=False,
                    ),
                )
            results.append(res)
            statuses.append(
                {
                    "name": call.name,
                    "status": "done" if res.ok else "error",
                    "latency_ms": res.latency_ms,
                    "call_id": call_id,
                    "error_type": res.error.type if res.error else None,
                }
            )

        for task in pending:
            meta = tasks[task]
            call = meta["call"]
            call_id = meta["call_id"]
            start_time = meta["start"]
            task.cancel()
            latency_ms = self._elapsed_ms(start_time)
            timeout_result = ToolResult(
                ok=False,
                source=call.name,
                as_of=None,
                latency_ms=latency_ms,
                warnings=[],
                data=None,
                error=ToolError(
                    type="timeout",
                    message=f"global timeout after {global_timeout_s}s",
                    retryable=True,
                ),
            )
            results.append(timeout_result)
            statuses.append(
                {
                    "name": call.name,
                    "status": "timeout",
                    "latency_ms": latency_ms,
                    "call_id": call_id,
                    "error_type": "timeout",
                }
            )

        return results, statuses

    async def _invoke_tool(
        self,
        call: ToolCallSpec,
        ctx: ToolContext,
        budgets: Dict[str, ToolBudget],
    ) -> ToolResult:
        spec = self._registry.get(call.name)
        if not spec:
            return ToolResult(
                ok=False,
                source=call.name,
                as_of=None,
                latency_ms=0,
                warnings=[],
                data=None,
                error=ToolError(
                    type="unknown_tool",
                    message=f"Tool {call.name} not registered",
                    retryable=False,
                ),
            )

        budget = budgets.get(call.name, ToolBudget())
        args, cap_error = self._apply_caps(call, budget)
        if cap_error:
            return cap_error

        try:
            validated = spec.input_model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                source=call.name,
                as_of=None,
                latency_ms=0,
                warnings=[],
                data=None,
                error=ToolError(
                    type="invalid_args",
                    message=str(exc),
                    retryable=False,
                ),
            )

        start = time.perf_counter()
        try:
            output = await asyncio.wait_for(spec.run(validated, ctx), timeout=budget.timeout_s)
            latency_ms = self._elapsed_ms(start)
            return ToolResult(
                ok=True,
                source=call.name,
                as_of=now_iso(),
                latency_ms=latency_ms,
                warnings=[],
                data=output,
                error=None,
            )
        except asyncio.TimeoutError:
            latency_ms = self._elapsed_ms(start)
            return ToolResult(
                ok=False,
                source=call.name,
                as_of=None,
                latency_ms=latency_ms,
                warnings=[],
                data=None,
                error=ToolError(
                    type="timeout",
                    message=f"timeout after {budget.timeout_s}s",
                    retryable=True,
                ),
            )
        except Exception as exc:
            latency_ms = self._elapsed_ms(start)
            return ToolResult(
                ok=False,
                source=call.name,
                as_of=None,
                latency_ms=latency_ms,
                warnings=[],
                data=None,
                error=ToolError(
                    type="tool_error",
                    message=str(exc),
                    retryable=False,
                ),
            )

    def _apply_caps(self, call: ToolCallSpec, budget: ToolBudget) -> Tuple[Dict[str, Any], Optional[ToolResult]]:
        args = dict(call.arguments or {})
        if call.name == "get_portfolio_context":
            top_n = args.get("top_n", 8)
            if budget.max_items is not None and int(top_n) > budget.max_items:
                return args, self._cap_error(call.name, "top_n", budget.max_items)
            args["top_n"] = int(top_n)
        elif call.name == "get_fundamentals":
            symbols = args.get("symbols") or []
            if budget.max_symbols is not None and len(symbols) > budget.max_symbols:
                return args, self._cap_error(call.name, "symbols", budget.max_symbols)
        elif call.name == "get_sec_snippets":
            limit = args.get("limit", budget.max_snippets or 6)
            if budget.max_snippets is not None and int(limit) > budget.max_snippets:
                return args, self._cap_error(call.name, "limit", budget.max_snippets)
            args["limit"] = int(limit)
        elif call.name == "get_news":
            max_results = args.get("max_results", budget.max_results or 5)
            if budget.max_results is not None and int(max_results) > budget.max_results:
                return args, self._cap_error(call.name, "max_results", budget.max_results)
            args["max_results"] = int(max_results)
        elif call.name == "get_web_search":
            max_results = args.get("max_results", budget.max_results or 5)
            if budget.max_results is not None and int(max_results) > budget.max_results:
                return args, self._cap_error(call.name, "max_results", budget.max_results)
            args["max_results"] = int(max_results)
        return args, None

    def _cap_error(self, tool_name: str, field: str, cap: int) -> ToolResult:
        return ToolResult(
            ok=False,
            source=tool_name,
            as_of=None,
            latency_ms=0,
            warnings=[],
            data=None,
            error=ToolError(
                type="cap_exceeded",
                message=f"{field} exceeds cap {cap}",
                retryable=False,
            ),
        )

    async def _maybe_holdings_snapshot(
        self,
        calls: List[ToolCallSpec],
        ctx: ToolContext,
    ) -> Optional[Dict[str, Any]]:
        if ctx.db is None or ctx.finnhub is None or ctx.user_id is None:
            return None
        if not any(call.name == "get_portfolio_context" for call in calls):
            return None
        top_n = 0
        for call in calls:
            if call.name != "get_portfolio_context":
                continue
            top_n = max(top_n, int(call.arguments.get("top_n", 0) or 0))
        try:
            return await get_holdings_with_live_prices(
                user_id=str(ctx.user_id),
                db=ctx.db,
                finnhub=ctx.finnhub,
                currency=ctx.user_currency or "USD",
                top_only=False,
                top_n=top_n,
                include_weights=True,
            )
        except Exception as exc:
            logger.warning("holdings snapshot failed: %s", exc)
            return None

    @staticmethod
    def _elapsed_ms(start: Optional[float]) -> int:
        if not start:
            return 0
        return int((time.perf_counter() - start) * 1000)
