import unittest
from services.ai.chat.finnhub_tools import FinnhubToolRegistry


class _FakeFinnhubService:
    async def get_price(self, symbol: str, typ: str = "stock"):
        return {
            "symbol": symbol,
            "formattedSymbol": symbol,
            "currentPrice": 100.5,
            "previousClose": 99.0,
            "high": 101.0,
            "low": 98.7,
            "open": 99.5,
            "currency": "USD",
        }

    async def fetch_profile(self, symbol: str, client=None):
        return {"name": "Apple Inc", "exchange": "NASDAQ", "finnhubIndustry": "Technology"}

    async def fetch_basic_financials(self, symbol: str, client=None):
        return {"metric": {"peTTM": 25.0, "roeTTM": 15.0}}

    async def fetch_peers(self, symbol: str, client=None):
        return ["MSFT", "GOOGL"]


async def _run_invalid_tool_args():
    reg = FinnhubToolRegistry(service=_FakeFinnhubService())
    return await reg.execute("get_quote", {})


async def _run_quote_success():
    reg = FinnhubToolRegistry(service=_FakeFinnhubService())
    return await reg.execute("get_quote", {"symbol": "aapl"})


class TestFinnhubTools(unittest.TestCase):
    def test_execute_invalid_tool_args(self):
        import asyncio

        res = asyncio.run(_run_invalid_tool_args())
        self.assertFalse(res.ok)
        self.assertIn("Invalid tool arguments", res.error or "")

    def test_execute_quote_success(self):
        import asyncio

        res = asyncio.run(_run_quote_success())
        self.assertTrue(res.ok)
        self.assertEqual(res.data["symbol"], "AAPL")
        self.assertEqual(res.data["currentPrice"], 100.5)


if __name__ == "__main__":
    unittest.main()
