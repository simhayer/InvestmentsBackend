import asyncio
import json
import os
import unittest
from unittest.mock import patch

from services.ai.chat.chat_models import ChatRequest
from services.ai.chat.chat_orchestrator import ChatOrchestrator
from services.ai.chat.intent_parser import IntentDecision, ToolCall


class _FakeGeminiConfig:
    model = "fake-model"


class _FakeGeminiClient:
    def __init__(self):
        self.config = _FakeGeminiConfig()

    async def stream_answer(self, *, system_prompt: str, user_prompt: str, allow_web_search=None):
        for chunk in ["Hello ", "world"]:
            yield chunk

    async def quick_generate(self, *, system_prompt: str, user_prompt: str, model_override=None):
        return "Hello there"


class _FakeToolResult:
    def __init__(self, tool_name: str, ok: bool = True, data=None, data_gaps=None, error=None):
        self.tool_name = tool_name
        self.ok = ok
        self.data = data or {}
        self.data_gaps = data_gaps or []
        self.error = error

    def model_dump(self):
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "data_gaps": self.data_gaps,
            "error": self.error,
        }


class _FakeTools:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def execute(self, tool_name: str, arguments):
        self.calls.append((tool_name, arguments))
        response = self._responses.get(tool_name)
        if callable(response):
            return response(arguments)
        if response is None:
            return _FakeToolResult(tool_name=tool_name, ok=True, data={"echo": arguments})
        return response


class _FakeIntentParser:
    def __init__(self, decisions):
        self._decisions = decisions
        self.calls = 0

    async def parse(self, *, conversation_text: str, context_text: str, last_user: str, request_id=None):
        idx = self.calls
        self.calls += 1
        if idx >= len(self._decisions):
            return self._decisions[-1]
        return self._decisions[idx]


def _decision(
    *,
    intent: str = "general_finance",
    action: str = "answer",
    tool_name=None,
    arguments=None,
    tools=None,
    reason: str = "ok",
    use_web: bool = False,
) -> IntentDecision:
    return IntentDecision(
        intent=intent,  # type: ignore[arg-type]
        use_web=use_web,
        action=action,  # type: ignore[arg-type]
        tools=tools or [],
        tool_name=tool_name,
        arguments=arguments or {},
        reason=reason,
    )


def _parse_event(raw: str):
    lines = [ln for ln in raw.split("\n") if ln]
    event = lines[0].replace("event: ", "")
    payload = json.loads(lines[1].replace("data: ", ""))
    return event, payload


async def _collect_events(orchestrator: ChatOrchestrator, prompt: str = "what is aapl price?"):
    req = ChatRequest(messages=[{"role": "user", "content": prompt}], conversation_id="cid-1")
    out = []
    async for event in orchestrator.stream_sse(req):
        out.append(_parse_event(event))
    return out


class TestChatOrchestrator(unittest.TestCase):
    def test_orchestrator_stream_with_legacy_single_tool(self):
        parser = _FakeIntentParser(
            [
                _decision(
                    intent="quote_lookup",
                    action="tool",
                    tool_name="get_quote",
                    arguments={"symbol": "AAPL"},
                    reason="live quote",
                )
            ]
        )
        tools = _FakeTools(
            {
                "get_quote": _FakeToolResult(
                    tool_name="get_quote", ok=True, data={"symbol": "AAPL", "currentPrice": 101.2}
                )
            }
        )
        orch = ChatOrchestrator(gemini_client=_FakeGeminiClient(), tools=tools, intent_parser=parser)
        events = asyncio.run(_collect_events(orch))

        names = [name for name, _ in events]
        self.assertIn("tool_call", names)
        self.assertIn("tool_result", names)
        self.assertIn("token", names)
        self.assertEqual(names[-1], "done")
        self.assertEqual(parser.calls, 1)

    def test_orchestrator_stream_with_parallel_tool_batch(self):
        with patch.dict(os.environ, {"CHAT_ENABLE_MULTI_TOOL_BATCH": "1"}, clear=False):
            parser = _FakeIntentParser(
                [
                    _decision(
                        intent="symbol_analysis",
                        action="tool",
                        tools=[
                            ToolCall(tool_name="get_quote", arguments={"symbol": "AAPL"}),
                            ToolCall(tool_name="get_basic_financials", arguments={"symbol": "AAPL"}),
                        ],
                        reason="price and fundamentals",
                    )
                ]
            )
            tools = _FakeTools(
                {
                    "get_quote": _FakeToolResult(
                        tool_name="get_quote", ok=True, data={"symbol": "AAPL", "currentPrice": 101.2}
                    ),
                    "get_basic_financials": _FakeToolResult(
                        tool_name="get_basic_financials", ok=True, data={"peTTM": 22.1}
                    ),
                }
            )
            orch = ChatOrchestrator(gemini_client=_FakeGeminiClient(), tools=tools, intent_parser=parser)
            events = asyncio.run(_collect_events(orch))

        tool_calls = [payload for name, payload in events if name == "tool_call"]
        tool_results = [payload for name, payload in events if name == "tool_result"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(len(tool_results), 2)
        self.assertEqual(parser.calls, 1)

    def test_batch_partial_failure_still_returns_answer(self):
        with patch.dict(os.environ, {"CHAT_ENABLE_MULTI_TOOL_BATCH": "1"}, clear=False):
            parser = _FakeIntentParser(
                [
                    _decision(
                        intent="symbol_analysis",
                        action="tool",
                        tools=[
                            ToolCall(tool_name="get_symbol_analysis", arguments={"symbol": "AAPL"}),
                            ToolCall(tool_name="get_quote", arguments={"symbol": "AAPL"}),
                        ],
                        reason="deep plus price",
                    )
                ]
            )
            tools = _FakeTools(
                {
                    "get_symbol_analysis": _FakeToolResult(
                        tool_name="get_symbol_analysis",
                        ok=False,
                        data={},
                        data_gaps=["Analysis unavailable"],
                        error="Analysis unavailable",
                    ),
                    "get_quote": _FakeToolResult(
                        tool_name="get_quote", ok=True, data={"symbol": "AAPL", "currentPrice": 101.2}
                    ),
                    "get_basic_financials": _FakeToolResult(
                        tool_name="get_basic_financials", ok=True, data={"peTTM": 22.1}
                    ),
                }
            )
            orch = ChatOrchestrator(gemini_client=_FakeGeminiClient(), tools=tools, intent_parser=parser)
            events = asyncio.run(_collect_events(orch))

        names = [name for name, _ in events]
        tool_results = [payload for name, payload in events if name == "tool_result"]
        self.assertIn("token", names)
        self.assertEqual(names[-1], "done")
        self.assertTrue(any(item.get("ok") is False for item in tool_results))
        self.assertTrue(any(item.get("tool_name") == "get_basic_financials" for item in tool_results))


if __name__ == "__main__":
    unittest.main()
