import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from schemas.stock_report import StockReport
from services.synthesis.stock_report_synthesizer import synthesize_stock_report


class StockReportSynthesizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizer_returns_valid_report(self) -> None:
        fake_report = {
            "symbol": "AAPL",
            "as_of": "2025-01-01T00:00:00Z",
            "quick_take": "Test summary.",
            "what_changed_recently": [],
            "fundamentals_snapshot": {
                "market_cap": None,
                "pe_ttm": None,
                "revenue_growth_yoy": None,
                "gross_margin": None,
                "operating_margin": None,
                "free_cash_flow": None,
                "debt_to_equity": None,
                "summary": "Limited data.",
            },
            "catalysts_next_30_90d": [],
            "risks": [],
            "sentiment": {
                "overall": "neutral",
                "drivers": [],
                "sources": [],
            },
            "scenarios": {
                "bull": {
                    "thesis": "Bull case placeholder.",
                    "key_assumptions": [],
                    "watch_items": [],
                    "sources": [],
                },
                "base": {
                    "thesis": "Base case placeholder.",
                    "key_assumptions": [],
                    "watch_items": [],
                    "sources": [],
                },
                "bear": {
                    "thesis": "Bear case placeholder.",
                    "key_assumptions": [],
                    "watch_items": [],
                    "sources": [],
                },
            },
            "confidence": {
                "score_0_100": 42,
                "rationale": "Test rationale.",
            },
            "citations": [],
            "data_gaps": [],
        }
        inputs = {
            "symbol": "AAPL",
            "as_of": "2025-01-01T00:00:00Z",
            "fundamentals": {},
            "news": [],
            "filings": [],
            "data_gaps": [],
        }

        async_mock = AsyncMock(return_value=SimpleNamespace(final_output=fake_report))
        with patch(
            "services.synthesis.stock_report_synthesizer.Runner.run",
            new=async_mock,
        ):
            report = await synthesize_stock_report(inputs, timeout_s=1.0)

        self.assertIsInstance(report, StockReport)
        self.assertEqual(report.symbol, "AAPL")
        self.assertEqual(report.quick_take, "Test summary.")
        self.assertEqual(report.model_dump(), fake_report)
