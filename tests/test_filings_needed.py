import unittest

from services.filings.tavily_filings import needs_filings_for_request


class FilingsNeededTests(unittest.TestCase):
    def test_explicit_flag(self) -> None:
        self.assertTrue(needs_filings_for_request(None, True))

    def test_keyword_match(self) -> None:
        request = "Please summarize the latest 10-K and any material event filings."
        self.assertTrue(needs_filings_for_request(request, False))

    def test_no_match(self) -> None:
        request = "Give me a quick overview of recent news."
        self.assertFalse(needs_filings_for_request(request, False))
