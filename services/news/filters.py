from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from services.news.ranking import EVENT_KEYWORDS, get_domain_tier, has_event_keyword

Classification = Literal[
    "tier1_news",
    "tier2_news",
    "press_release",
    "filing",
    "analysis_low",
    "quote_junk",
    "options_junk",
]

QUOTE_DOMAIN_PATTERNS = {
    "robinhood.com": ["/stocks/"],
    "finance.yahoo.com": ["/quote/"],
    "ca.finance.yahoo.com": ["/quote/"],
    "cnn.com": ["/markets/stocks"],
    "marketwatch.com": ["/investing/stock"],
    "nasdaq.com": ["/market-activity/stocks"],
    "tradingview.com": ["/symbols/"],
    "tipranks.com": ["/stocks/"],
    "investing.com": ["/equities/"],
    "seekingalpha.com": ["/symbol/"],
}

LOW_SIGNAL_DOMAINS = {
    "finviz.com",
    "marketbeat.com",
    "stocktwits.com",
    "reddit.com",
}

OPTION_PATH_KEYWORDS = ("options", "option-chain", "calls", "puts")

JUNK_KEYWORDS = (
    "stock price",
    "quote",
    "chart",
    "historical prices",
    "options",
    "option chain",
    "calls",
    "puts",
    "dividend history",
    "earnings date",
    "technical analysis",
    "insider trading",
    "forecast",
    "target price",
)

FILING_HINTS = ("8-k", "10-q", "10-k", "sec.gov/archives")


def extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return domain[4:] if domain.startswith("www.") else domain


def is_option_chain(url: str, title: str) -> bool:
    lower_title = (title or "").lower()
    if any(word in lower_title for word in OPTION_PATH_KEYWORDS):
        return True
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(word in path for word in OPTION_PATH_KEYWORDS)


def is_quote_page(url: str, title: str) -> bool:
    parsed = urlparse(url)
    domain = extract_domain(url)
    path = parsed.path.lower()
    if domain in QUOTE_DOMAIN_PATTERNS:
        if any(fragment in path for fragment in QUOTE_DOMAIN_PATTERNS[domain]):
            return True
    lower_title = (title or "").lower()
    return "stock price" in lower_title or "quote" in lower_title


def _has_junk_keyword(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in JUNK_KEYWORDS)


def _is_filing(url: str) -> bool:
    lowered = (url or "").lower()
    return any(hint in lowered for hint in FILING_HINTS)


def _event_exception(domain_tier: str, title: str) -> bool:
    if domain_tier not in {"tier1", "tier2", "press_release"}:
        return False
    return any(keyword in (title or "").lower() for keyword in EVENT_KEYWORDS)


def classify_source(url: str, title: str) -> Classification:
    domain = extract_domain(url)
    domain_tier = get_domain_tier(domain)

    if _is_filing(url):
        return "filing"

    if is_option_chain(url, title):
        return "options_junk"

    if is_quote_page(url, title):
        return "quote_junk"

    if _has_junk_keyword(title) or _has_junk_keyword(url):
        if domain_tier == "press_release" and "earnings" in (title or "").lower():
            return "press_release"
        if _event_exception(domain_tier, title):
            return "tier1_news" if domain_tier == "tier1" else "tier2_news"
        return "quote_junk"

    if domain in LOW_SIGNAL_DOMAINS:
        return "analysis_low"

    if domain_tier == "press_release":
        return "press_release"
    if domain_tier == "tier1":
        return "tier1_news"
    if domain_tier == "tier2":
        if domain == "seekingalpha.com" and not has_event_keyword(title):
            return "analysis_low"
        return "tier2_news"

    return "analysis_low"


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"https://{url}")
    scheme = "https"
    netloc = parsed.netloc.lower()
    netloc = netloc[4:] if netloc.startswith("www.") else netloc
    path = re.sub(r"/+$", "", parsed.path)
    query = parsed.query
    if query:
        params = []
        for pair in query.split("&"):
            key = pair.split("=", 1)[0].lower()
            if key.startswith("utm_"):
                continue
            if key in {"gclid", "fbclid", "mc_cid", "mc_eid"}:
                continue
            params.append(pair)
        query = "&".join(params)
    rebuilt = f"{scheme}://{netloc}{path}"
    if query:
        rebuilt = f"{rebuilt}?{query}"
    return rebuilt
