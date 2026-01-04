from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

TIER1_DOMAINS = {
    "reuters.com",
    "ft.com",
    "wsj.com",
    "bloomberg.com",
    "sec.gov",
    "prnewswire.com",
    "businesswire.com",
}

TIER2_DOMAINS = {
    "cnbc.com",
    "theverge.com",
    "nytimes.com",
    "barrons.com",
    "seekingalpha.com",
    "fool.com",
    "benzinga.com",
}

PRESS_RELEASE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
}

LOW_SIGNAL_DOMAINS = {
    "stocktwits.com",
    "reddit.com",
    "finviz.com",
    "marketbeat.com",
    "tipranks.com",
}

DOWNRANK_DOMAINS = {
    "benzinga.com",
    "fool.com",
}

EVENT_KEYWORDS = {
    "earnings",
    "guidance",
    "raises",
    "cuts",
    "acquisition",
    "merger",
    "partnership",
    "sec",
    "8-k",
    "10-q",
    "10-k",
    "lawsuit",
    "settlement",
    "investigation",
    "downgrade",
    "upgrade",
    "price target",
    "regulator",
    "antitrust",
    "recall",
}

GENERIC_KEYWORDS = {
    "stock price",
    "quote",
    "chart",
    "forecast",
    "technical",
    "options",
    "call",
    "put",
}


def is_ir_domain(domain: str) -> bool:
    if not domain:
        return False
    lowered = domain.lower()
    return "investor" in lowered or "investors" in lowered or lowered.startswith("ir.")


def get_domain_tier(domain: str) -> str:
    if not domain:
        return "low_signal"
    if domain in PRESS_RELEASE_DOMAINS or is_ir_domain(domain):
        return "press_release"
    if domain in TIER1_DOMAINS:
        return "tier1"
    if domain in TIER2_DOMAINS:
        return "tier2"
    if domain in LOW_SIGNAL_DOMAINS:
        return "low_signal"
    return "tier2"


def has_event_keyword(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in EVENT_KEYWORDS)


def has_generic_keyword(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in GENERIC_KEYWORDS)


def recency_score(published_at: datetime | None, recency_days: int) -> float:
    if not published_at or recency_days <= 0:
        return 0.05
    now = datetime.now(timezone.utc)
    age_days = (now - published_at).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    if age_days > recency_days:
        return 0.0
    return 0.6 * (1.0 - (age_days / recency_days))


def base_score(source_tier: str, classification: str) -> float:
    if classification == "filing":
        return 0.8
    if source_tier == "tier1":
        return 1.0
    if source_tier == "tier2":
        return 0.7
    if source_tier == "press_release":
        return 0.6
    return 0.3


def score_item(
    *,
    title: str,
    source_domain: str,
    source_tier: str,
    classification: str,
    published_at: datetime | None,
    recency_days: int,
) -> float:
    score = base_score(source_tier, classification)
    score += recency_score(published_at, recency_days)

    if has_event_keyword(title):
        score += 0.2
    if has_generic_keyword(title):
        score -= 0.2

    if source_domain in DOWNRANK_DOMAINS:
        score -= 0.15
    if not published_at:
        score -= 0.25

    return score


def best_source_tier(domain: str) -> str:
    return get_domain_tier(domain)


def rank_key(
    *,
    score: float,
    published_at: datetime | None,
    order: int,
) -> tuple[float, float, int]:
    timestamp = published_at.timestamp() if published_at else 0.0
    return (score, timestamp, -order)


def filter_domains(domains: Iterable[str], *, limit: int = 4) -> list[str]:
    cleaned: list[str] = []
    for domain in domains:
        if domain and domain not in cleaned:
            cleaned.append(domain)
        if len(cleaned) >= limit:
            break
    return cleaned
