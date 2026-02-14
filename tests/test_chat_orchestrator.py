import asyncio
import json
import unittest

from services.ai.chat.chat_models import ChatRequest
from services.ai.chat.chat_orchestrator import ChatOrchestrator


class _FakeGeminiConfig:
    model = "fake-model"


class _FakeGeminiClient:
    def __init__(self, decision):
        self.config = _FakeGeminiConfig()
        self._decision = decision

    async def decide_tool_action(self, *, system_prompt: str, user_prompt: str, model_override=None):
        return self._decision

    async def stream_answer(self, *, system_prompt: str, user_prompt: str, allow_web_search=None):
        for chunk in ["Hello ", "world"]:
            yield chunk


class _FakeToolResult:
    def __init__(self):
        self.tool_name = "get_quote"
        self.ok = True
        self.data = {"symbol": "AAPL", "currentPrice": 101.2}
        self.data_gaps = []
        self.error = None

    def model_dump(self):
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "data_gaps": self.data_gaps,
            "error": self.error,
        }


class _FakeTools:
    async def execute(self, tool_name: str, arguments):
        return _FakeToolResult()


def _parse_event(raw: str):
    lines = [ln for ln in raw.split("\n") if ln]
    event = lines[0].replace("event: ", "")
    payload = json.loads(lines[1].replace("data: ", ""))
    return event, payload


async def _collect_events(orchestrator: ChatOrchestrator, prompt: str = "what is aapl price?"):
    req = ChatRequest(messages=[{"role": "user", "content": prompt}], conversation_id="cid-1")
    out = []
    async for event in orchestrator.stream_sse(req):
        out.append(event)
    return out


class TestChatOrchestrator(unittest.TestCase):
    def test_orchestrator_stream_with_tool(self):
        decision = {"action": "tool", "tool_name": "get_quote", "arguments": {"symbol": "AAPL"}, "reason": "live quote"}
        orch = ChatOrchestrator(gemini_client=_FakeGeminiClient(decision), finnhub_tools=_FakeTools())
        events = asyncio.run(_collect_events(orch))

        parsed = [_parse_event(e)[0] for e in events]
        self.assertIn("meta", parsed)
        self.assertIn("tool_call", parsed)
        self.assertIn("tool_result", parsed)
        self.assertIn("token", parsed)
        self.assertEqual(parsed[-1], "done")

    def test_orchestrator_stream_without_tool(self):
        decision = {"action": "answer", "tool_name": None, "arguments": {}, "reason": "general question"}
        orch = ChatOrchestrator(gemini_client=_FakeGeminiClient(decision), finnhub_tools=_FakeTools())
        events = asyncio.run(_collect_events(orch, prompt="hello there"))
        parsed = [_parse_event(e)[0] for e in events]
        self.assertNotIn("tool_call", parsed)
        self.assertEqual(parsed[-1], "done")


if __name__ == "__main__":
    unittest.main()
