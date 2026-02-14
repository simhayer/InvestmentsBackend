from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

from services.ai.chat.chat_models import ChatRequest, format_sse
from services.ai.chat.chat_prompts import build_finance_system_prompt
from services.ai.chat.finnhub_tools import FinnhubToolRegistry
from services.ai.chat.gemini_stream_client import GeminiStreamClient
from services.ai.chat.intent_parser import IntentParser

logger = logging.getLogger(__name__)

DisconnectFn = Callable[[], Awaitable[bool]]

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
        finnhub_tools: Optional[FinnhubToolRegistry] = None,
        intent_parser: Optional[IntentParser] = None,
        max_tool_rounds: int = 2,
        decision_timeout_s: float = 2.5,
        tool_timeout_s: float = 8.0,
    ):
        self.gemini = gemini_client or GeminiStreamClient()
        self.tools = finnhub_tools or FinnhubToolRegistry()
        self.intent_parser = intent_parser or IntentParser(self.gemini, timeout_s=decision_timeout_s)
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.decision_timeout_s = float(decision_timeout_s)
        self.tool_timeout_s = float(tool_timeout_s)

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

    def _context_text(self, req: ChatRequest) -> str:
        if not req.context:
            return ""
        data = req.context.model_dump(exclude_none=True)
        if not data:
            return ""
        return json.dumps(data, ensure_ascii=True)

    async def _safe_is_disconnected(self, fn: Optional[DisconnectFn]) -> bool:
        if fn is None:
            return False
        try:
            return bool(await fn())
        except Exception:
            return False

    def _small_talk_response(self) -> str:
        return (
            "I can help with stock/crypto questions, portfolio analysis, and market updates. "
            "Tell me what you want to analyze."
        )

    async def stream_sse(
        self,
        req: ChatRequest,
        *,
        is_disconnected: Optional[DisconnectFn] = None,
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        req_id = req.conversation_id or f"chat-{int(start * 1000)}"
        tool_payload: Dict[str, Any] = {}

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
            step_start = time.perf_counter()
            conversation_text = self._conversation_text(req)
            context_text = self._context_text(req)
            computed_allow_web = req.allow_web_search
            _trace_info(
                "chat.context_ready req_id=%s elapsed_ms=%s convo_chars=%s ctx_chars=%s",
                req_id,
                int((time.perf_counter() - step_start) * 1000),
                len(conversation_text),
                len(context_text),
            )

            yield format_sse(
                "meta",
                {
                    "request_id": req_id,
                    "policy_mode": "llm_router",
                },
            )
            _trace_info("chat.meta.policy_sent req_id=%s", req_id)

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

            # Guardrail: portfolio-intent queries should fetch user portfolio context.
            if decision.intent == "portfolio_guidance" and decision.action != "tool":
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
                _trace_info("chat.portfolio_guardrail req_id=%s tool=%s", req_id, decision.tool_name)

            # Intent-aware fast path for small-talk.
            if decision.intent == "small_talk":
                _trace_info("chat.small_talk_fast_path req_id=%s", req_id)
                quick = self._small_talk_response()
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

            rounds = 0
            while rounds < self.max_tool_rounds:
                rounds += 1
                action = decision.action
                tool_name = decision.tool_name
                arguments = decision.arguments or {}
                reason = decision.reason or ""
                _trace_info(
                    "chat.tool_loop req_id=%s round=%s action=%s tool=%s",
                    req_id,
                    rounds,
                    action,
                    tool_name,
                )

                if action != "tool" or not tool_name:
                    _trace_info("chat.tool_skip req_id=%s reason=no_tool_action", req_id)
                    break

                if await self._safe_is_disconnected(is_disconnected):
                    _trace_info("chat.disconnect req_id=%s at_tool_call", req_id)
                    return

                yield format_sse(
                    "tool_call",
                    {"request_id": req_id, "tool_name": tool_name, "arguments": arguments, "reason": reason},
                )
                _trace_info("chat.tool_call_sent req_id=%s tool=%s", req_id, tool_name)

                step_start = time.perf_counter()
                tool_result = await asyncio.wait_for(
                    self.tools.execute(str(tool_name), arguments),
                    timeout=self.tool_timeout_s,
                )
                _trace_info(
                    "chat.tool_result_ready req_id=%s tool=%s elapsed_ms=%s ok=%s",
                    req_id,
                    tool_name,
                    int((time.perf_counter() - step_start) * 1000),
                    tool_result.ok,
                )
                tool_payload = tool_result.model_dump()

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
                # Single tool execution is enough for v1; preserve round guard for future extension.
                break

            if await self._safe_is_disconnected(is_disconnected):
                _trace_info("chat.disconnect req_id=%s before_stream", req_id)
                return

            # Step 2: final streamed answer
            final_prompt = (
                f"Conversation:\n{conversation_text}\n\n"
                f"Optional app context:\n{context_text or 'None'}\n\n"
                f"Tool result (if any):\n{json.dumps(tool_payload, ensure_ascii=True)}\n\n"
                "Answer the latest user request with clear practical guidance."
            )
            system_prompt = build_finance_system_prompt()

            # Initial keepalive so frontend knows stream is active.
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
            _trace_info("chat.done req_id=%s elapsed_ms=%s", req_id, elapsed_ms)
            yield format_sse(
                "done",
                {
                    "request_id": req_id,
                    "elapsed_ms": elapsed_ms,
                    "finish_reason": "stop",
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
