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
from services.ai.chat.tool_registry import ChatToolRegistry

logger = logging.getLogger(__name__)

DisconnectFn = Callable[[], Awaitable[bool]]

_DIRECT_TOOL_INTENTS = {"portfolio_lookup"}


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
        decision_timeout_s: float = 2.5,
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
            },
        )

        try:
            # ── Step 1: Build system prompt with page context + investor profile ──
            step_start = time.perf_counter()
            if self._db and self._user_id:
                system_prompt = build_system_prompt(
                    db=self._db,
                    user_id=self._user_id,
                    req=req,
                )
            else:
                from services.ai.chat.chat_prompts import build_finance_system_prompt
                system_prompt = build_finance_system_prompt()

            conversation_text = self._conversation_text(req)
            context_text = build_context_text(req)
            computed_allow_web = req.allow_web_search
            _trace_info(
                "chat.context_ready req_id=%s elapsed_ms=%s convo_chars=%s ctx_chars=%s",
                req_id,
                int((time.perf_counter() - step_start) * 1000),
                len(conversation_text),
                len(context_text),
            )

            # ── Step 2: Page acknowledgment ──
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

            # ── Step 3: Intent classification ──
            yield format_sse(
                "meta",
                {"request_id": req_id, "policy_mode": "llm_router"},
            )

            step_start = time.perf_counter()
            last_user = self._last_user_message(req)
            decision = await self.intent_parser.parse(
                conversation_text=conversation_text,
                context_text=context_text,
                last_user=last_user,
                request_id=req_id,
            )
            _trace_info(
                "chat.intent_ready req_id=%s elapsed_ms=%s intent=%s action=%s tool=%s use_web=%s",
                req_id,
                int((time.perf_counter() - step_start) * 1000),
                decision.intent,
                decision.action,
                decision.tool_name,
                decision.use_web,
            )

            # Guardrail: portfolio-intent queries should fetch portfolio data
            if decision.intent in {"portfolio_lookup", "portfolio_guidance", "portfolio_analysis"} and decision.action != "tool":
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
                tool_name = decision.tool_name
                arguments = decision.arguments or {}
                reason = decision.reason or ""

                _trace_info(
                    "chat.tool_loop req_id=%s round=%s/%s action=%s tool=%s",
                    req_id,
                    rounds,
                    self.max_tool_rounds,
                    action,
                    tool_name,
                )

                if action != "tool" or not tool_name:
                    _trace_info("chat.tool_skip req_id=%s reason=no_tool_action", req_id)
                    break

                if not self.tools:
                    _trace_info("chat.tool_skip req_id=%s reason=no_tool_registry", req_id)
                    break

                # Don't retry a tool that already failed in this request
                if tool_name in failed_tools:
                    _trace_info(
                        "chat.tool_skip req_id=%s reason=already_failed tool=%s",
                        req_id,
                        tool_name,
                    )
                    break

                if await self._safe_is_disconnected(is_disconnected):
                    _trace_info("chat.disconnect req_id=%s at_tool_call round=%s", req_id, rounds)
                    return

                # Emit thinking event
                yield format_sse(
                    "thinking",
                    {
                        "request_id": req_id,
                        "text": f"Looking up {tool_name}...",
                        "round": rounds,
                    },
                )

                # Emit tool_call
                yield format_sse(
                    "tool_call",
                    {
                        "request_id": req_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "reason": reason,
                    },
                )
                _trace_info("chat.tool_call_sent req_id=%s tool=%s", req_id, tool_name)

                # Execute tool
                step_start = time.perf_counter()
                try:
                    tool_result = await asyncio.wait_for(
                        self.tools.execute(str(tool_name), arguments),
                        timeout=self.tool_timeout_s,
                    )
                except asyncio.TimeoutError:
                    _trace_info("chat.tool_timeout req_id=%s tool=%s", req_id, tool_name)
                    failed_tools.add(tool_name)
                    tool_result_dict = {
                        "ok": False,
                        "tool_name": tool_name,
                        "error": f"Tool {tool_name} timed out",
                        "data": {},
                        "data_gaps": ["timeout"],
                    }
                    tool_results.append(tool_result_dict)
                    yield format_sse(
                        "tool_result",
                        {"request_id": req_id, **tool_result_dict},
                    )
                    break

                _trace_info(
                    "chat.tool_result_ready req_id=%s tool=%s elapsed_ms=%s ok=%s",
                    req_id,
                    tool_name,
                    int((time.perf_counter() - step_start) * 1000),
                    tool_result.ok,
                )

                if not tool_result.ok:
                    failed_tools.add(tool_name)

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

                # Direct-tool intents: one tool call is enough, skip replan
                if decision.intent in _DIRECT_TOOL_INTENTS and tool_result.ok:
                    _trace_info(
                        "chat.replan_skip req_id=%s reason=direct_tool_intent intent=%s",
                        req_id, decision.intent,
                    )
                    break

                # Re-plan: ask the intent parser if we need another tool
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
                        "chat.replan req_id=%s round=%s elapsed_ms=%s action=%s tool=%s",
                        req_id,
                        rounds,
                        int((time.perf_counter() - step_start) * 1000),
                        decision.action,
                        decision.tool_name,
                    )
                    # Loop continues: if decision.action == "tool", next iteration executes it;
                    # if decision.action == "answer", the while condition check will break.

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

            final_prompt = (
                f"Conversation:\n{conversation_text}\n\n"
                f"Optional app context:\n{context_text or 'None'}\n\n"
                f"Tool results:\n{tool_payload_str}\n\n"
                "Answer the latest user request with clear practical guidance."
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
