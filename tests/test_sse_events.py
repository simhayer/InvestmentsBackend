import unittest

from agent.state_models import DataRequirementsPlan, GraphState, RequestContext, ToolResult
from services.ai.chat_agent.chat_agent_service import build_stream_events


class SSEEventsTests(unittest.TestCase):
    def test_stream_event_sequence(self):
        state = GraphState(
            message="test",
            user_id="u1",
            user_currency="USD",
            session_id="s1",
            trace_id="trace123",
            turn_id="turn123",
            request_context=RequestContext(intent="news_q", tickers=["AAPL"], needs_recency=True),
            data_requirements=DataRequirementsPlan(required_data=["news"], optional_data=[]),
            tool_results=[
                ToolResult(
                    ok=True,
                    source="get_news",
                    as_of=None,
                    latency_ms=5,
                    warnings=[],
                    data={"items": [{"title": "Headline"}]},
                    error=None,
                )
            ],
            tool_statuses=[{"name": "get_news", "status": "done", "latency_ms": 5, "error_type": None}],
        )
        events = build_stream_events(state, "Hello world", 12.5, "s1", chunk_size=5)
        event_types = [event["event"] for event in events]
        self.assertEqual(event_types[0], "plan")
        self.assertIn("tool_status", event_types)
        self.assertIn("delta", event_types)
        self.assertEqual(event_types[-1], "final")
        final_payload = events[-1]["data"]
        self.assertEqual(final_payload["trace_id"], "trace123")


if __name__ == "__main__":
    unittest.main()
