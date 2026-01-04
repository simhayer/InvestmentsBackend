import unittest

from services.filings.tavily_filings import normalize_filing_results
from services.news.tavily_news import normalize_news_results


class TavilyParsingTests(unittest.TestCase):
    def test_news_dedupe_and_recency(self) -> None:
        results = [
            {
                "title": "Older news",
                "url": "https://example.com/a",
                "content": "x" * 600,
                "published_date": "2024-01-02",
            },
            {
                "title": "Duplicate url",
                "url": "https://example.com/a",
                "content": "duplicate",
                "published_date": "2024-01-03",
            },
            {
                "title": "Newer news",
                "url": "https://example.com/b",
                "content": "short snippet",
                "published_date": "2024-01-05",
            },
        ]
        result = normalize_news_results(results, max_items=2, recency_days=3650, min_recent_items=0)
        items = result.items
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["url"], "https://example.com/b")
        self.assertEqual(items[0]["id"], "news_1")
        self.assertLessEqual(len(items[0]["snippet"]), 500)
        self.assertEqual(len(set(item["url"] for item in items)), 2)

    def test_filings_dedupe_and_limit(self) -> None:
        results = [
            {
                "title": "10-K filed",
                "url": "https://sec.gov/filing1",
                "content": "A filing summary",
                "published_date": "2024-02-01",
            },
            {
                "title": "10-K duplicate",
                "url": "https://sec.gov/filing1",
                "content": "duplicate",
                "published_date": "2024-02-02",
            },
            {
                "title": "8-K filed",
                "url": "https://sec.gov/filing2",
                "content": "Another filing summary",
                "published_date": "2024-03-01",
            },
        ]
        items = normalize_filing_results(results, max_items=1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "filing_1")
        self.assertEqual(items[0]["url"], "https://sec.gov/filing2")
