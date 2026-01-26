import unittest

from agent.decisioning_graph import parse_request_context


class IntentRoutingTests(unittest.TestCase):
    def test_portfolio_intent(self):
        ctx = parse_request_context("How is my portfolio doing today?")
        self.assertEqual(ctx.intent, "portfolio_q")
        self.assertTrue(ctx.needs_portfolio)
        self.assertTrue(ctx.needs_recency)

    def test_sec_intent_with_section(self):
        ctx = parse_request_context("Show the risk factors for NVDA")
        self.assertEqual(ctx.intent, "sec_q")
        self.assertIn("risk", ctx.requested_sections or [])

    def test_education_intent(self):
        ctx = parse_request_context("Explain what an ETF is")
        self.assertEqual(ctx.intent, "education_q")


if __name__ == "__main__":
    unittest.main()
