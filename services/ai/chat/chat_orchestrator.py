"""Chat orchestrator v2 — ReAct agentic loop.

Flow:
  1. Assemble system prompt (ContextAssembler — page ctx, investor profile, tool manifest)
  2. Emit ``page_ack`` if page context is present
  3. Intent parse → first tool decision
  4. ReAct tool loop (up to ``max_tool_rounds``):
       Execute tool → emit result → re-plan with accumulated results
  5. Stream final answer via Gemini
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from services.ai.chat.chat_models import ChatRequest, format_sse
from services.ai.chat.context_assembler import build_system_prompt, build_context_text
from services.ai.chat.gemini_stream_client import GeminiStreamClient
from services.ai.chat.intent_parser import IntentParser
from services.ai.chat.tool_registry import ChatToolRegistry, TOOL_TIMEOUTS, TOOL_FALLBACKS

logger = logging.getLogger(__name__)

DisconnectFn = Callable[[], Awaitable[bool]]

_DIRECT_TOOL_INTENTS = {"portfolio_lookup"}
_SINGLE_TOOL_INTENTS = {"quote_lookup", "company_profile", "fundamentals"}
_BUNDLE_TERMINAL_INTENTS = {"peer_comparison", "symbol_analysis", "risk_analysis"}


def _trace_info(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.info(msg, *args)


def _trace_exception(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.exception(msg, *args)


class ChatOrchestrator:
    def __init__(
        self,
        *,
        gemini_client: Optional[GeminiStreamClient] = None,
        tools: Optional[ChatToolRegistry] = None,
        intent_parser: Optional[IntentParser] = None,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
        max_tool_rounds: int = 5,
        decision_timeout_s: float = 4.0,
        tool_timeout_s: float = 15.0,
    ):
        self.gemini = gemini_client or GeminiStreamClient()
        self._small_talk_model = os.getenv("GEMINI_SMALL_TALK_MODEL") or self.gemini.config.model
        self.tools = tools
        self.intent_parser = intent_parser or IntentParser(
            self.gemini, timeout_s=decision_timeout_s
        )
        self._db = db
        self._user_id = user_id
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.decision_timeout_s = float(decision_timeout_s)
        self.tool_timeout_s = float(tool_timeout_s)
        self._enable_multi_tool_batch = (os.getenv("CHAT_ENABLE_MULTI_TOOL_BATCH") or "0") == "1"
        self._enable_speculative_prefetch = (os.getenv("CHAT_ENABLE_SPECULATIVE_PREFETCH") or "0") == "1"
        self._advice_style_v1 = (os.getenv("CHAT_ADVICE_STYLE_V1") or "0") == "1"

    # ── helpers ──────────────────────────────────────────────────────

    def _last_user_message(self, req: ChatRequest) -> str:
        for msg in reversed(req.messages):
            if msg.role == "user":
                return msg.content
        return req.messages[-1].content

    def _conversation_text(self, req: ChatRequest) -> str:
        lines = []
        for m in req.messages:
            lines.append(f"{m.role.upper()}: {m.content}")
        return "\n".join(lines)

    async def _safe_is_disconnected(self, fn: Optional[DisconnectFn]) -> bool:
        if fn is None:
            return False
        try:
            return bool(await fn())
        except Exception:
            return False

    _SMALL_TALK_FALLBACK = (
        "I can help with stock/crypto questions, portfolio analysis, and market updates. "
        "Tell me what you want to analyze."
    )

    async def _small_talk_response(self, last_user: str) -> str:
        from services.ai.chat.chat_prompts import SMALL_TALK_SYSTEM_PROMPT

        text = await self.gemini.quick_generate(
            system_prompt=SMALL_TALK_SYSTEM_PROMPT,
            user_prompt=last_user,
            model_override=self._small_talk_model,
        )
        return text or self._SMALL_TALK_FALLBACK

    def _tool_call_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        args_str = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=True, default=str)
        return f"{tool_name}:{args_str}"

    def _decision_tool_calls(self, decision: Any) -> List[Dict[str, Any]]:
        """Normalize new tools[] schema with legacy single-tool fallback."""
        calls: List[Dict[str, Any]] = []
        if self._enable_multi_tool_batch:
            raw_tools = getattr(decision, "tools", None) or []
            for entry in raw_tools:
                tool_name = getattr(entry, "tool_name", None)
                arguments = getattr(entry, "arguments", {})
                if not tool_name:
                    continue
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append({"tool_name": str(tool_name), "arguments": arguments})
        if not calls and getattr(decision, "tool_name", None):
            calls.append(
                {
                    "tool_name": str(decision.tool_name),
                    "arguments": decision.arguments or {},
                }
            )

        # Drop exact duplicates while preserving order.
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for call in calls:
            key = self._tool_call_key(call["tool_name"], call["arguments"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        return deduped

    async def _prefetch_quote_for_symbol(
        self,
        *,
        symbol: str,
        req_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.tools:
            return None
        normalized = (symbol or "").strip().upper()
        if not normalized:
            return None
        args = {"symbol": normalized}
        timeout = TOOL_TIMEOUTS.get("get_quote", self.tool_timeout_s)
        try:
            result = await asyncio.wait_for(
                self.tools.execute("get_quote", args),
                timeout=timeout,
            )
            _trace_info(
                "chat.prefetch_quote_ok req_id=%s symbol=%s ok=%s",
                req_id,
                normalized,
                result.ok,
            )
            return {"tool_name": "get_quote", "arguments": args, "result": result}
        except Exception as exc:
            _trace_info(
                "chat.prefetch_quote_skip req_id=%s symbol=%s err=%s",
                req_id,
                normalized,
                type(exc).__name__,
            )
            return None

    async def _run_fallback_tools(
        self,
        failed_tool: str,
        arguments: Dict[str, Any],
        failed_tools: set,
        req_id: str,
    ) -> List[Any]:
        """Run lighter Tier-1 tools in parallel when a heavy tool fails."""
        fallback_names = TOOL_FALLBACKS.get(failed_tool, [])
        to_run = [n for n in fallback_names if n not in failed_tools]
        if not to_run or not self.tools:
            return []
        _trace_info(
            "chat.fallback_start req_id=%s failed_tool=%s fallbacks=%s",
            req_id, failed_tool, to_run,
        )

        async def _one(name: str):
            timeout = TOOL_TIMEOUTS.get(name, self.tool_timeout_s)
            return await asyncio.wait_for(
                self.tools.execute(name, arguments), timeout=timeout,
            )

        raw = await asyncio.gather(
            *[_one(n) for n in to_run], return_exceptions=True,
        )
        results = []
        for fb_name, res in zip(to_run, raw):
            if isinstance(res, Exception):
                _trace_info(
                    "chat.fallback_error req_id=%s tool=%s err=%s",
                    req_id, fb_name, type(res).__name__,
                )
                continue
            _trace_info(
                "chat.fallback_ok req_id=%s tool=%s ok=%s",
                req_id, fb_name, res.ok,
            )
            results.append(res)
        return results

    # ── main SSE stream ─────────────────────────────────────────────

    async def stream_sse(
        self,
        req: ChatRequest,
        *,
        is_disconnected: Optional[DisconnectFn] = None,
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        req_id = req.conversation_id or f"chat-{int(start * 1000)}"
        tool_results: List[Dict[str, Any]] = []

        _trace_info("chat.start req_id=%s messages=%s", req_id, len(req.messages))
        yield format_sse(
            "meta",
            {
                "request_id": req_id,
                "model": self.gemini.config.model,
                "max_tool_rounds": self.max_tool_rounds,
                "feature_flags": {
                    "multi_tool_batch": self._enable_multi_tool_batch,
                    "speculative_prefetch": self._enable_speculative_prefetch,
                    "advice_style_v1": self._advice_style_v1,
                },
            },
        )
        _trace_info(
            "chat.flags req_id=%s multi_tool_batch=%s speculative_prefetch=%s advice_style_v1=%s",
            req_id,
            self._enable_multi_tool_batch,
            self._enable_speculative_prefetch,
            self._advice_style_v1,
        )

        try:
            # ── Steps 1-3: Context + intent in parallel ──
            step_start = time.perf_counter()
            conversation_text = self._conversation_text(req)
            context_text = build_context_text(req)
            last_user = self._last_user_message(req)
            computed_allow_web = req.allow_web_search

            # Page acknowledgment (immediate, no LLM needed)
            page = req.context.page if req.context else None
            if page:
                ack_text = f"Looking at {page.page_type} page"
                if page.symbol:
                    ack_text = f"Looking at {page.symbol}"
                yield format_sse(
                    "page_ack",
                    {"request_id": req_id, "text": ack_text},
                )
                _trace_info("chat.page_ack req_id=%s text=%s", req_id, ack_text)

            yield format_sse(
                "meta",
                {"request_id": req_id, "policy_mode": "llm_router"},
            )

            # Build system prompt (DB query) and parse intent (LLM call) concurrently
            async def _build_prompt() -> str:
                if self._db and self._user_id:
                    return await asyncio.to_thread(
                        build_system_prompt,
                        db=self._db,
                        user_id=self._user_id,
                        req=req,
                    )
                from services.ai.chat.chat_prompts import build_finance_system_prompt
                return build_finance_system_prompt()

            prefetched_first_tool: Optional[Dict[str, Any]] = None
            prefetch_task: Optional[asyncio.Task] = None
            if (
                self._enable_speculative_prefetch
                and self.tools
                and page
                and page.symbol
            ):
                prefetch_task = asyncio.create_task(
                    self._prefetch_quote_for_symbol(symbol=page.symbol, req_id=req_id)
                )

            if prefetch_task is not None:
                system_prompt, decision, prefetched_first_tool = await asyncio.gather(
                    _build_prompt(),
                    self.intent_parser.parse(
                        conversation_text=conversation_text,
                        context_text=context_text,
                        last_user=last_user,
                        request_id=req_id,
                    ),
                    prefetch_task,
                )
            else:
                system_prompt, decision = await asyncio.gather(
                    _build_prompt(),
                    self.intent_parser.parse(
                        conversation_text=conversation_text,
                        context_text=context_text,
                        last_user=last_user,
                        request_id=req_id,
                    ),
                )
            _trace_info(
                "chat.context_and_intent_ready req_id=%s elapsed_ms=%s convo_chars=%s ctx_chars=%s intent=%s action=%s tool=%s tools_count=%s use_web=%s prefetched=%s",
                req_id,
                int((time.perf_counter() - step_start) * 1000),
                len(conversation_text),
                len(context_text),
                decision.intent,
                decision.action,
                decision.tool_name,
                len(getattr(decision, "tools", []) or []),
                decision.use_web,
                bool(prefetched_first_tool),
            )

            # Guardrail: portfolio-intent queries should fetch portfolio data
            if decision.intent in {"portfolio_lookup", "portfolio_guidance", "portfolio_analysis"} and (
                decision.action != "tool" or not decision.tool_name
            ):
                preferred_currency = None
                if req.context and req.context.preferred_currency:
                    preferred_currency = req.context.preferred_currency.strip().upper()
                args: Dict[str, Any] = {"top_n": 8}
                if preferred_currency in {"USD", "CAD"}:
                    args["currency"] = preferred_currency
                decision.action = "tool"
                decision.tool_name = "get_portfolio_overview"
                decision.arguments = args
                decision.reason = "portfolio_guardrail_auto_tool"
                _trace_info(
                    "chat.portfolio_guardrail req_id=%s tool=%s", req_id, decision.tool_name
                )

            # Fast path: small-talk (dynamic via lite model)
            if decision.intent == "small_talk":
                _trace_info("chat.small_talk_fast_path req_id=%s model=%s", req_id, self._small_talk_model)
                quick = await self._small_talk_response(last_user)
                yield format_sse("heartbeat", {"request_id": req_id})
                yield format_sse("token", {"request_id": req_id, "text": quick})
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                yield format_sse(
                    "done",
                    {
                        "request_id": req_id,
                        "elapsed_ms": elapsed_ms,
                        "finish_reason": "small_talk_fast_path",
                    },
                )
                return

            if req.allow_web_search is None:
                computed_allow_web = bool(decision.use_web)

            yield format_sse(
                "meta",
                {
                    "request_id": req_id,
                    "allow_web_search": bool(computed_allow_web),
                    "router_reason": decision.reason or "",
                    "intent": decision.intent,
                },
            )

            # ── Step 4: ReAct tool loop ──
            rounds = 0
            failed_tools: set[str] = set()  # avoid retrying the same broken tool
            while rounds < self.max_tool_rounds:
                rounds += 1
                action = decision.action
                reason = decision.reason or ""
                tool_calls = self._decision_tool_calls(decision)
                first_tool_name = tool_calls[0]["tool_name"] if tool_calls else None

                _trace_info(
                    "chat.tool_loop req_id=%s round=%s/%s action=%s first_tool=%s tools_count=%s",
                    req_id,
                    rounds,
                    self.max_tool_rounds,
                    action,
                    first_tool_name,
                    len(tool_calls),
                )

                if action != "tool" or not tool_calls:
                    _trace_info("chat.tool_skip req_id=%s reason=no_tool_action", req_id)
                    break

                if not self.tools:
                    _trace_info("chat.tool_skip req_id=%s reason=no_tool_registry", req_id)
                    break

                executable_calls: List[Dict[str, Any]] = []
                for call in tool_calls:
                    name = call["tool_name"]
                    if name in failed_tools:
                        _trace_info(
                            "chat.tool_skip req_id=%s reason=already_failed tool=%s",
                            req_id,
                            name,
                        )
                        continue
                    executable_calls.append(call)
                if not executable_calls:
                    _trace_info("chat.tool_skip req_id=%s reason=no_executable_tools", req_id)
                    break

                if await self._safe_is_disconnected(is_disconnected):
                    _trace_info("chat.disconnect req_id=%s at_tool_call round=%s", req_id, rounds)
                    return

                tool_label = ", ".join(call["tool_name"] for call in executable_calls[:3])
                yield format_sse(
                    "thinking",
                    {
                        "request_id": req_id,
                        "text": f"Looking up {tool_label}...",
                        "round": rounds,
                    },
                )

                for call in executable_calls:
                    yield format_sse(
                        "tool_call",
                        {
                            "request_id": req_id,
                            "tool_name": call["tool_name"],
                            "arguments": call["arguments"],
                            "reason": reason,
                        },
                    )
                    _trace_info("chat.tool_call_sent req_id=%s tool=%s", req_id, call["tool_name"])

                async def _run_one_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                    started_at = time.perf_counter()
                    timeout = TOOL_TIMEOUTS.get(name, self.tool_timeout_s)
                    try:
                        result = await asyncio.wait_for(
                            self.tools.execute(name, arguments),
                            timeout=timeout,
                        )
                        return {
                            "kind": "result",
                            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                            "result": result,
                        }
                    except asyncio.TimeoutError:
                        return {
                            "kind": "timeout",
                            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                            "timeout_s": timeout,
                        }
                    except Exception as exc:
                        return {
                            "kind": "error",
                            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                            "error": exc,
                        }

                outcomes: Dict[str, Dict[str, Any]] = {}
                pending_calls: List[Dict[str, Any]] = []
                pending_tasks: List[Any] = []
                for call in executable_calls:
                    key = self._tool_call_key(call["tool_name"], call["arguments"])
                    if (
                        rounds == 1
                        and prefetched_first_tool
                        and call["tool_name"] == prefetched_first_tool.get("tool_name")
                        and self._tool_call_key(
                            call["tool_name"], call["arguments"]
                        )
                        == self._tool_call_key(
                            str(prefetched_first_tool.get("tool_name", "")),
                            prefetched_first_tool.get("arguments", {}),
                        )
                    ):
                        outcomes[key] = {
                            "kind": "result",
                            "elapsed_ms": 0,
                            "result": prefetched_first_tool["result"],
                            "prefetched": True,
                        }
                        _trace_info(
                            "chat.prefetch_reuse req_id=%s tool=%s",
                            req_id,
                            call["tool_name"],
                        )
                        prefetched_first_tool = None
                        continue
                    pending_calls.append(call)
                    pending_tasks.append(_run_one_tool(call["tool_name"], call["arguments"]))

                if pending_tasks:
                    raw_outcomes = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for call, outcome in zip(pending_calls, raw_outcomes):
                        key = self._tool_call_key(call["tool_name"], call["arguments"])
                        if isinstance(outcome, Exception):
                            outcomes[key] = {"kind": "error", "elapsed_ms": 0, "error": outcome}
                        else:
                            outcomes[key] = outcome

                batch_had_success = False
                for call in executable_calls:
                    tool_name = call["tool_name"]
                    arguments = call["arguments"]
                    key = self._tool_call_key(tool_name, arguments)
                    outcome = outcomes.get(key, {"kind": "error", "error": RuntimeError("missing outcome")})
                    kind = outcome.get("kind")

                    if kind == "timeout":
                        failed_tools.add(tool_name)
                        timeout_s = float(outcome.get("timeout_s") or TOOL_TIMEOUTS.get(tool_name, self.tool_timeout_s))
                        _trace_info(
                            "chat.tool_timeout req_id=%s tool=%s timeout_s=%.1f",
                            req_id,
                            tool_name,
                            timeout_s,
                        )
                        tool_result_dict = {
                            "ok": False,
                            "tool_name": tool_name,
                            "error": f"Tool {tool_name} timed out",
                            "data": {},
                            "data_gaps": ["timeout"],
                        }
                        tool_results.append(tool_result_dict)
                        yield format_sse("tool_result", {"request_id": req_id, **tool_result_dict})
                        for fb_res in await self._run_fallback_tools(
                            tool_name, arguments, failed_tools, req_id,
                        ):
                            fb_dict = fb_res.model_dump()
                            tool_results.append(fb_dict)
                            yield format_sse(
                                "tool_result",
                                {
                                    "request_id": req_id,
                                    "tool_name": fb_res.tool_name,
                                    "ok": fb_res.ok,
                                    "data": fb_res.data,
                                    "data_gaps": fb_res.data_gaps,
                                    "error": fb_res.error,
                                },
                            )
                        continue

                    if kind == "error":
                        failed_tools.add(tool_name)
                        err = outcome.get("error")
                        _trace_info(
                            "chat.tool_error req_id=%s tool=%s err=%s",
                            req_id,
                            tool_name,
                            type(err).__name__ if err else "UnknownError",
                        )
                        tool_result_dict = {
                            "ok": False,
                            "tool_name": tool_name,
                            "error": f"Tool {tool_name} failed",
                            "data": {},
                            "data_gaps": ["execution_error"],
                        }
                        tool_results.append(tool_result_dict)
                        yield format_sse("tool_result", {"request_id": req_id, **tool_result_dict})
                        for fb_res in await self._run_fallback_tools(
                            tool_name, arguments, failed_tools, req_id,
                        ):
                            fb_dict = fb_res.model_dump()
                            tool_results.append(fb_dict)
                            yield format_sse(
                                "tool_result",
                                {
                                    "request_id": req_id,
                                    "tool_name": fb_res.tool_name,
                                    "ok": fb_res.ok,
                                    "data": fb_res.data,
                                    "data_gaps": fb_res.data_gaps,
                                    "error": fb_res.error,
                                },
                            )
                        continue

                    tool_result = outcome["result"]
                    _trace_info(
                        "chat.tool_result_ready req_id=%s tool=%s elapsed_ms=%s ok=%s prefetched=%s",
                        req_id,
                        tool_name,
                        int(outcome.get("elapsed_ms") or 0),
                        tool_result.ok,
                        bool(outcome.get("prefetched")),
                    )

                    if tool_result.ok:
                        batch_had_success = True
                    else:
                        failed_tools.add(tool_name)
                        for fb_res in await self._run_fallback_tools(
                            tool_name, arguments, failed_tools, req_id,
                        ):
                            fb_dict = fb_res.model_dump()
                            tool_results.append(fb_dict)
                            yield format_sse(
                                "tool_result",
                                {
                                    "request_id": req_id,
                                    "tool_name": fb_res.tool_name,
                                    "ok": fb_res.ok,
                                    "data": fb_res.data,
                                    "data_gaps": fb_res.data_gaps,
                                    "error": fb_res.error,
                                },
                            )

                    result_dict = tool_result.model_dump()
                    tool_results.append(result_dict)
                    yield format_sse(
                        "tool_result",
                        {
                            "request_id": req_id,
                            "tool_name": tool_result.tool_name,
                            "ok": tool_result.ok,
                            "data": tool_result.data,
                            "data_gaps": tool_result.data_gaps,
                            "error": tool_result.error,
                        },
                    )

                # Skip extra re-plan calls for deterministic intents.
                if batch_had_success and decision.intent in (_DIRECT_TOOL_INTENTS | _SINGLE_TOOL_INTENTS):
                    _trace_info(
                        "chat.replan_skip req_id=%s reason=single_tool_intent intent=%s",
                        req_id,
                        decision.intent,
                    )
                    break
                if (
                    batch_had_success
                    and len(executable_calls) > 1
                    and decision.intent in _BUNDLE_TERMINAL_INTENTS
                ):
                    _trace_info(
                        "chat.replan_skip req_id=%s reason=deterministic_bundle intent=%s tools=%s",
                        req_id,
                        decision.intent,
                        len(executable_calls),
                    )
                    break

                # Re-plan once per batch to decide whether another round is needed.
                if rounds < self.max_tool_rounds:
                    tool_ctx = json.dumps(tool_results, ensure_ascii=True, default=str)
                    replan_context = f"{context_text}\n\nTool results so far:\n{tool_ctx}"
                    step_start = time.perf_counter()
                    decision = await self.intent_parser.parse(
                        conversation_text=conversation_text,
                        context_text=replan_context,
                        last_user=last_user,
                        request_id=req_id,
                    )
                    _trace_info(
                        "chat.replan req_id=%s round=%s elapsed_ms=%s action=%s tool=%s tools_count=%s",
                        req_id,
                        rounds,
                        int((time.perf_counter() - step_start) * 1000),
                        decision.action,
                        decision.tool_name,
                        len(getattr(decision, "tools", []) or []),
                    )

            # ── Step 5: Stream final answer ──
            if await self._safe_is_disconnected(is_disconnected):
                _trace_info("chat.disconnect req_id=%s before_stream", req_id)
                return

            # Direct-tool intents get a focused prompt that avoids unsolicited analysis
            if decision.intent in _DIRECT_TOOL_INTENTS:
                from services.ai.chat.chat_prompts import DIRECT_TOOL_ANSWER_PROMPT
                system_prompt = DIRECT_TOOL_ANSWER_PROMPT
                computed_allow_web = False

            tool_payload_str = json.dumps(tool_results, ensure_ascii=True, default=str) if tool_results else "{}"
            final_instruction = "Answer the latest user request with clear practical guidance."
            if self._advice_style_v1:
                final_instruction = (
                    "Answer the latest user request with practical, advisor-like guidance. "
                    "Give a clear takeaway first, include key risks or caveats, and provide 1-2 concrete next steps "
                    "when the user asks for recommendations. Calibrate detail to the investor profile when present."
                )

            final_prompt = (
                f"Conversation:\n{conversation_text}\n\n"
                f"Optional app context:\n{context_text or 'None'}\n\n"
                f"Tool results:\n{tool_payload_str}\n\n"
                f"{final_instruction}"
            )

            # Keepalive
            yield format_sse("heartbeat", {"request_id": req_id})
            _trace_info("chat.heartbeat_sent req_id=%s", req_id)

            first_token_at: Optional[float] = None
            async for chunk in self.gemini.stream_answer(
                system_prompt=system_prompt,
                user_prompt=final_prompt,
                allow_web_search=bool(computed_allow_web),
            ):
                if await self._safe_is_disconnected(is_disconnected):
                    _trace_info("chat.disconnect req_id=%s during_stream", req_id)
                    return
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    _trace_info(
                        "chat.first_token req_id=%s first_token_ms=%s",
                        req_id,
                        int((first_token_at - start) * 1000),
                    )
                yield format_sse("token", {"request_id": req_id, "text": chunk})

            citations: List[Dict[str, str]] = []
            try:
                pop_citations = getattr(self.gemini, "pop_last_citations", None)
                if callable(pop_citations):
                    raw_citations = pop_citations() or []
                    if isinstance(raw_citations, list):
                        citations = [
                            c for c in raw_citations
                            if isinstance(c, dict) and isinstance(c.get("url"), str)
                        ]
            except Exception:
                citations = []

            if citations:
                for item in citations:
                    payload = {"request_id": req_id, "url": item.get("url")}
                    if item.get("title"):
                        payload["title"] = item.get("title")
                    yield format_sse("citation", payload)
                _trace_info(
                    "chat.citations req_id=%s count=%s",
                    req_id,
                    len(citations),
                )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            _trace_info(
                "chat.done req_id=%s elapsed_ms=%s tool_rounds=%s",
                req_id,
                elapsed_ms,
                len(tool_results),
            )
            yield format_sse(
                "done",
                {
                    "request_id": req_id,
                    "elapsed_ms": elapsed_ms,
                    "finish_reason": "stop",
                    "tool_rounds": len(tool_results),
                    "citations": citations,
                },
            )
        except asyncio.TimeoutError:
            _trace_exception("chat.timeout req_id=%s", req_id)
            yield format_sse(
                "error",
                {"request_id": req_id, "message": "Request timed out. Please retry."},
            )
        except Exception:
            _trace_exception("chat.error req_id=%s", req_id)
            yield format_sse(
                "error",
                {"request_id": req_id, "message": "Unable to process chat request right now."},
            )
