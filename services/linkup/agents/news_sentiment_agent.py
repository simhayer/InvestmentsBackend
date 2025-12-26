from __future__ import annotations
from services.linkup.linkup_search import linkup_structured_search
from services.linkup.schemas.news_sentiment_schema import NEWS_SENTIMENT_SCHEMA

def build_news_sentiment_query(base_currency, symbols):
    return {
        "role": (
            "You are a professional portfolio analyst using Linkup as your research engine. "
            "You must output ONLY valid JSON that matches the NEWS_SENTIMENT_SCHEMA. "
            "You are not allowed to give trading instructions or recommendations."
        ),
        "step_1_task": [
            "Scan recent, credible sources for news and events related to the provided symbols and major macro drivers.",
            "Summarize only well-supported headlines and catalysts; avoid fabricating dates, numbers, or company actions.",
            "Identify portfolio-relevant risks and monitoring points (company-specific and macro).",
            "Provide an aggregated sentiment view (bullish/neutral/bearish) with narrative drivers.",
        ],
        "step_2_context": [
            "Assume North American markets (NASDAQ, NYSE, TSX) as the primary context.",
            "Use Linkup search results and Yahoo Finance-style fundamentals where available.",
            "If there is little or no recent news for a symbol, prefer omitting it over guessing.",
        ],
        "step_3_references": [
            "Use only credible sources returned by Linkup.",
            "When summarizing a source, include inline citations in the form 【source†L#-L#】 in text fields.",
            "Do NOT invent headlines, event dates, or percentage moves. If uncertain, state 'unclear' or omit the item.",
        ],
        "step_4_evaluate": [
            "Check that each headline and catalyst could realistically be supported by at least one source.",
            "If support is weak or ambiguous, lower the relevant section_confidence and mention it in explainability.limitations.",
            "Avoid any language that tells the user to buy, sell, add, trim, or hold positions.",
        ],
        "step_5_iterate": [
            "After drafting, remove any speculative statements that are not clearly supported by sources.",
            "Prefer fewer, well-supported developments over many low-confidence ones.",
        ],
        "constraints": [
            "Output must be STRICTLY valid JSON per NEWS_SENTIMENT_SCHEMA.",
            "Populate assets_affected ONLY with symbols from the provided list.",
            "All content is informational and descriptive, not investment advice.",
            "Do NOT use imperative verbs like 'buy', 'sell', 'add', 'trim', or 'hold'.",
        ],
        "portfolio_context": {
            "base_currency": base_currency,
            "symbols": symbols,
        },
    }


def call_link_up_for_news(base_currency, symbols, days_of_news=7):
    return linkup_structured_search(
        query_obj=build_news_sentiment_query(base_currency=base_currency, symbols=symbols),
        schema=NEWS_SENTIMENT_SCHEMA,
        days=days_of_news,
        include_sources=False,
        depth="standard",
        max_retries=2,
    )