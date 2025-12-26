from services.linkup.schemas.market_summary_schema import MARKET_SUMMARY_SCHEMA
from services.linkup.linkup_search import linkup_structured_search

def get_linkup_market_summary() -> dict:
    return linkup_structured_search(
        query_obj={
            "role": (
                "You are a financial analyst. Summarize today's key US market developments "
                "in a 'cause → impact' format. For each event, explain what happened, why it happened, "
                "and how it affected financial markets. Focus on recent data and news from the past week "
                "(S&P 500, NASDAQ, Dow Jones, yields, CPI, Fed policy, energy, etc.). "
                "Use reliable sources like Bloomberg, Reuters, and WSJ."
            )
        },
        schema=MARKET_SUMMARY_SCHEMA,
        days=7,
        include_sources=False,
        depth="deep",
        max_retries=2,
    )