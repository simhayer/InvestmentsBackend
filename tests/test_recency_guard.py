import unittest

from agent.decisioning_graph import evaluate_recency
from agent.state_models import ToolResult


class RecencyGuardTests(unittest.TestCase):
    def test_recency_missing_news(self):
        tool_results = []
        self.assertTrue(evaluate_recency(True, tool_results))

    def test_recency_with_news_items(self):
        tool_results = [
            ToolResult(
                ok=True,
                source="get_news",
                as_of=None,
                latency_ms=1,
                warnings=[],
                data={"items": [{"title": "Headline"}]},
                error=None,
            )
        ]
        self.assertFalse(evaluate_recency(True, tool_results))


if __name__ == "__main__":
    unittest.main()
