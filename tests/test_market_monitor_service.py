import asyncio
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from services.market_monitor_service import get_market_monitor_panel
from services.market_monitor_service import get_personalized_market_monitor_panel


class _FailingLLM:
    async def generate_json(self, *, system: str, user: str):
        raise RuntimeError("LLM unavailable")


class _FakeLLMService:
    async def generate_json(self, *, system: str, user: str):
        return {
            "cards": [
                {
                    "title": "Rates Pressure",
                    "summary": "Treasury-sensitive assets remain exposed to rate repricing.",
                    "signal": "bearish",
                    "time_horizon": "1-3d",
                },
                {
                    "title": "Crypto Bid",
                    "summary": "Digital assets are holding relative strength versus broader risk assets.",
                    "signal": "bullish",
                    "time_horizon": "intraday",
                },
            ]
        }


class TestMarketMonitorService(unittest.TestCase):
    def test_monitor_panel_builds_expected_sections(self):
        async def _run():
            with patch("services.market_monitor_service.get_global_brief_cached") as brief_mock, patch(
                "services.market_monitor_service.get_predictions_cached"
            ) as predictions_mock, patch(
                "services.market_monitor_service.get_global_news"
            ) as news_mock, patch(
                "services.market_monitor_service.get_llm_service",
                return_value=_FakeLLMService(),
            ), patch(
                "services.market_monitor_service._get_market_overview_data"
            ) as overview_mock:
                brief_mock.return_value = {
                    "as_of": "2026-03-02T00:00:00Z",
                    "market": "Global Markets",
                    "outlook": "Cautiously risk-on.",
                    "sections": [
                        {
                            "headline": "Fed & Rates",
                            "cause": "Bond markets are repricing policy expectations.",
                            "impact": "Duration stays sensitive to hawkish surprises.",
                        }
                    ],
                }
                predictions_mock.return_value = {
                    "as_of": "2026-03-02T00:01:00Z",
                    "outlook": "Cautiously risk-on with focus on rates.",
                }

                async def _news_side_effect(*, category: str, limit: int):
                    return [
                        {
                            "title": f"{category} headline",
                            "url": f"https://example.com/{category}",
                            "source": "Example",
                            "published_at": "2026-03-02T00:02:00Z",
                            "snippet": f"{category} summary",
                            "image": None,
                        }
                    ]

                news_mock.side_effect = _news_side_effect
                overview_mock.return_value = {
                    "fetched_at": "2026-03-02T00:03:00Z",
                    "items": [
                        {"key": "SPX", "label": "S&P 500", "price": 5100, "changePct": 0.6, "currency": "USD"},
                        {"key": "DJI", "label": "Dow Jones", "price": 39000, "changePct": -0.2, "currency": "USD"},
                        {"key": "BTC", "label": "BTC/USD", "price": 87000, "changePct": 1.8, "currency": "USD"},
                        {"key": "VIX", "label": "VIX", "price": 15.3, "changePct": -3.0, "currency": "USD"},
                    ],
                }

                return await get_market_monitor_panel(object(), force_refresh=True)

        payload = asyncio.run(_run())

        self.assertEqual(payload["title"], "Global Finance Monitor")
        self.assertEqual(payload["as_of"], "2026-03-02T00:01:00Z")
        self.assertEqual(payload["sections"]["world_brief"]["market"], "Global Markets")
        self.assertEqual(len(payload["sections"]["ai_insights"]), 2)
        self.assertEqual(len(payload["sections"]["news_streams"]), 4)
        self.assertEqual(payload["sections"]["market_pulse"][0]["key"], "major_indices")

    def test_monitor_panel_falls_back_when_ai_insights_fail(self):
        async def _run():
            with patch("services.market_monitor_service.get_global_brief_cached") as brief_mock, patch(
                "services.market_monitor_service.get_predictions_cached"
            ) as predictions_mock, patch(
                "services.market_monitor_service.get_global_news"
            ) as news_mock, patch(
                "services.market_monitor_service.get_llm_service",
                return_value=_FailingLLM(),
            ), patch(
                "services.market_monitor_service._get_market_overview_data"
            ) as overview_mock:
                brief_mock.return_value = {
                    "as_of": "2026-03-02T00:00:00Z",
                    "market": "Global Markets",
                    "outlook": None,
                    "sections": [
                        {
                            "headline": "Geopolitical Risk",
                            "cause": "Sanctions headlines and shipping disruption are pressuring energy-sensitive sectors.",
                            "impact": "Expect near-term volatility in cyclicals.",
                        },
                        {
                            "headline": "Crypto Resilience",
                            "cause": "Crypto is outperforming broader risk assets.",
                            "impact": "Momentum remains stronger than equities.",
                        },
                    ],
                }
                predictions_mock.return_value = {
                    "as_of": "2026-03-02T00:01:00Z",
                    "outlook": "Risk-off tone around geopolitics.",
                }

                async def _news_side_effect(*, category: str, limit: int):
                    return []

                news_mock.side_effect = _news_side_effect
                overview_mock.return_value = {
                    "fetched_at": "2026-03-02T00:03:00Z",
                    "items": [
                        {"key": "SPX", "label": "S&P 500", "price": 5100, "changePct": -0.9, "currency": "USD"},
                        {"key": "BTC", "label": "BTC/USD", "price": 87000, "changePct": 1.8, "currency": "USD"},
                    ],
                }

                return await get_market_monitor_panel(object(), force_refresh=True)

        payload = asyncio.run(_run())

        insights = payload["sections"]["ai_insights"]
        self.assertGreaterEqual(len(insights), 2)
        self.assertEqual(insights[0]["title"], "Market Regime")
        self.assertEqual(insights[0]["signal"], "bearish")

    def test_personalized_panel_uses_portfolio_context_by_default(self):
        async def _run():
            with patch(
                "services.market_monitor_service.get_market_monitor_panel"
            ) as base_mock, patch(
                "services.market_monitor_service._get_live_holdings_payload"
            ) as holdings_mock, patch(
                "services.market_monitor_service._get_portfolio_summary_data"
            ) as summary_mock, patch(
                "services.market_monitor_service._get_portfolio_inline_insights_data"
            ) as insights_mock, patch(
                "services.market_monitor_service.get_company_news_for_symbols"
            ) as news_mock:
                base_mock.return_value = {
                    "title": "Global Finance Monitor",
                    "sections": {"ai_insights": [], "world_brief": {}, "market_pulse": [], "news_streams": []},
                }
                holdings_mock.return_value = {
                    "top_items": [
                        {
                            "symbol": "AAPL",
                            "name": "Apple Inc",
                            "weight": 18.2,
                            "current_value": 18200,
                            "unrealized_pl_pct": 12.1,
                            "current_price": 210.0,
                            "currency": "USD",
                        },
                        {
                            "symbol": "MSFT",
                            "name": "Microsoft",
                            "weight": 15.5,
                            "current_value": 15500,
                            "unrealized_pl_pct": 8.0,
                            "current_price": 420.0,
                            "currency": "USD",
                        },
                    ]
                }
                summary_mock.return_value = {
                    "positions_count": 7,
                    "market_value": 100000,
                    "day_pl": 1200,
                    "day_pl_pct": 1.2,
                    "currency": "USD",
                }
                insights_mock.return_value = {
                    "healthBadge": "HHI 1400 - Well Diversified",
                    "performanceNote": "+14.0% vs SPY +11.0%",
                    "riskFlag": "Top 3 = 48% of portfolio",
                    "topPerformer": "AAPL +22%",
                    "actionNeeded": "Large-cap tech remains the main concentration to review.",
                }
                news_mock.return_value = {
                    "AAPL": [{"title": "Apple expands AI rollout", "url": "https://example.com/aapl", "source": "Example", "published_at": "2026-03-02T00:00:00Z", "snippet": "New device features launch.", "image": None}],
                    "MSFT": [{"title": "Microsoft cloud growth holds", "url": "https://example.com/msft", "source": "Example", "published_at": "2026-03-02T00:00:00Z", "snippet": "Azure remains resilient.", "image": None}],
                }
                return await get_personalized_market_monitor_panel(
                    object(),
                    user_id="42",
                    finnhub=object(),
                    currency="USD",
                    force_refresh=True,
                )

        payload = asyncio.run(_run())
        personalization = payload["personalization"]

        self.assertEqual(personalization["scope"], "portfolio")
        self.assertEqual(personalization["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(personalization["portfolio_snapshot"]["positions_count"], 7)
        self.assertGreaterEqual(len(personalization["insight_cards"]), 3)
        self.assertEqual(personalization["focus_news"][0]["symbol"], "AAPL")
        self.assertIsNone(personalization["watchlist"])

    def test_personalized_panel_supports_persisted_watchlist(self):
        async def _run():
            with patch(
                "services.market_monitor_service.get_market_monitor_panel"
            ) as base_mock, patch(
                "services.market_monitor_service._get_watchlist_context"
            ) as watchlist_mock, patch(
                "services.market_monitor_service.get_company_news_for_symbols"
            ) as news_mock:
                base_mock.return_value = {
                    "title": "Global Finance Monitor",
                    "sections": {"ai_insights": [], "world_brief": {}, "market_pulse": [], "news_streams": []},
                }
                watchlist_mock.return_value = (
                    {"id": 8, "name": "AI Leaders", "is_default": True},
                    ["NVDA", "TSLA"],
                )
                news_mock.return_value = {
                    "NVDA": [{"title": "Nvidia demand stays elevated", "url": "https://example.com/nvda", "source": "Example", "published_at": "2026-03-02T00:00:00Z", "snippet": "AI server orders remain strong.", "image": None}],
                    "TSLA": [{"title": "Tesla cuts incentives", "url": "https://example.com/tsla", "source": "Example", "published_at": "2026-03-02T00:00:00Z", "snippet": "Margin pressure remains in focus.", "image": None}],
                }
                return await get_personalized_market_monitor_panel(
                    object(),
                    user_id="42",
                    finnhub=object(),
                    currency="USD",
                    force_refresh=True,
                    watchlist_id=8,
                )

        payload = asyncio.run(_run())
        personalization = payload["personalization"]

        self.assertEqual(personalization["scope"], "watchlist")
        self.assertEqual(personalization["symbols"], ["NVDA", "TSLA"])
        self.assertIsNone(personalization["portfolio_snapshot"])
        self.assertEqual(personalization["watchlist"]["id"], 8)
        self.assertEqual(personalization["focus_news"][1]["symbol"], "TSLA")
        self.assertTrue(any(card["title"] == "NVDA Watch" for card in personalization["insight_cards"]))

    def test_personalized_panel_falls_back_when_user_has_no_holdings(self):
        async def _run():
            with patch(
                "services.market_monitor_service.get_market_monitor_panel"
            ) as base_mock, patch(
                "services.market_monitor_service._get_live_holdings_payload"
            ) as holdings_mock:
                base_mock.return_value = {
                    "title": "Global Finance Monitor",
                    "sections": {"ai_insights": [], "world_brief": {}, "market_pulse": [], "news_streams": []},
                }
                holdings_mock.return_value = {"top_items": []}
                return await get_personalized_market_monitor_panel(
                    object(),
                    user_id="42",
                    finnhub=object(),
                    currency="USD",
                    force_refresh=True,
                )

        payload = asyncio.run(_run())
        personalization = payload["personalization"]

        self.assertEqual(personalization["scope"], "global_fallback")
        self.assertEqual(personalization["symbols"], [])
        self.assertIsNotNone(personalization["empty_state"])


if __name__ == "__main__":
    unittest.main()
