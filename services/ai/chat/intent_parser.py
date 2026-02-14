from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, Literal, Optional

from services.ai.chat.chat_prompts import INTENT_ROUTER_SYSTEM_PROMPT, build_tool_manifest_prompt
from services.ai.chat.gemini_stream_client import GeminiStreamClient

logger = logging.getLogger(__name__)

def _trace_info(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.info(msg, *args)


def _trace_warning(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.warning(msg, *args)

IntentType = Literal[
    "small_talk",
    "quote_lookup",
    "company_profile",
    "fundamentals",
    "peer_comparison",
    "macro_news",
    "portfolio_guidance",
    "general_finance",
]


@dataclass
class IntentDecision:
    intent: IntentType
    use_web: bool
    action: Literal["tool", "answer"]
    tool_name: Optional[str]
    arguments: Dict[str, Any]
    reason: str


class IntentParser:
    def __init__(self, gemini: GeminiStreamClient, timeout_s: float = 2.5):
        self.gemini = gemini
        self.timeout_s = float(timeout_s)
        self.intent_model = os.getenv("GEMINI_INTENT_MODEL") or "gemini-2.0-flash-lite"

    async def parse(
        self,
        *,
        conversation_text: str,
        context_text: str,
        last_user: str,
        request_id: Optional[str] = None,
    ) -> IntentDecision:
        started = time.perf_counter()
        _trace_info(
            "intent.parse.start req_id=%s timeout_s=%.2f model=%s last_user_len=%s",
            request_id,
            self.timeout_s,
            self.intent_model,
            len(last_user or ""),
        )
        user_prompt = (
            f"{build_tool_manifest_prompt()}\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            f"Optional app context:\n{context_text or 'None'}\n\n"
            f"Latest user request:\n{last_user}"
        )
        try:
            raw = await asyncio.wait_for(
                self.gemini.decide_tool_action(
                    system_prompt=INTENT_ROUTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model_override=self.intent_model,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _trace_warning("intent.parse.timeout req_id=%s elapsed_ms=%s", request_id, elapsed_ms)
            return IntentDecision(
                intent="general_finance",
                use_web=False,
                action="answer",
                tool_name=None,
                arguments={},
                reason="intent_timeout",
            )

        intent = str((raw or {}).get("intent") or "general_finance")
        if intent not in {
            "small_talk",
            "quote_lookup",
            "company_profile",
            "fundamentals",
            "peer_comparison",
            "macro_news",
            "portfolio_guidance",
            "general_finance",
        }:
            intent = "general_finance"

        action = str((raw or {}).get("action") or "answer").lower()
        if action not in {"tool", "answer"}:
            action = "answer"

        tool_name = (raw or {}).get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = None

        args = (raw or {}).get("arguments")
        if not isinstance(args, dict):
            args = {}

        decision = IntentDecision(
            intent=intent,  # type: ignore[arg-type]
            use_web=bool((raw or {}).get("use_web", False)),
            action=action,  # type: ignore[arg-type]
            tool_name=tool_name,
            arguments=args,
            reason=str((raw or {}).get("reason") or "ok"),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _trace_info(
            "intent.parse.done req_id=%s elapsed_ms=%s intent=%s action=%s tool=%s use_web=%s reason=%s",
            request_id,
            elapsed_ms,
            decision.intent,
            decision.action,
            decision.tool_name,
            decision.use_web,
            decision.reason,
        )
        return decision
