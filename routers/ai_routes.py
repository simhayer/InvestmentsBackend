from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from services.auth_service import get_current_user
from services.ai_service import analyze_investment_portfolio, analyze_investment_symbol_perplexity, analyze_portfolio_perplexity
from services.holding_service import get_all_holdings
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from fastapi import APIRouter, Query
from services.portfolio_summary import summarize_portfolio_news
from services.finnhub_news_service import get_company_news_for_symbols
from services.helpers.linkup.symbol_analysis import get_linkup_symbol_analysis
from services.helpers.linkup.portfolio_analysis import get_portfolio_ai_layers_from_quotes
from services.yahoo_service import get_full_stock_data_many
from typing import List

router = APIRouter()

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Fetch holdings for this user
    holdings = get_all_holdings(user.id, db)
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings found for this user")

    # 2. Extract symbols
    symbols = [h.symbol for h in holdings if h.symbol]
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid symbols found")

    # 3. Bulk Yahoo fetch (fast, cached)
    quotes_map = get_full_stock_data_many(symbols)

    # 4. Call Linkup AI
    ai_layers = get_portfolio_ai_layers_from_quotes(
        quotes_map=quotes_map,
        base_currency="CAD",
        days_of_news=7,
        include_sources=True,
        timeout=60,
        targets={"Equities": 60, "Bonds": 30, "Cash": 10},  # optional
        symbols_preferred_order=[h.symbol for h in holdings],  # keeps consistent symbol tags
    )

    # 5. Return AI-only layer (or merge later on frontend/backend)
    return {"status": "ok", "user_id": user.id, "ai_layers": ai_layers}

class SymbolReq(BaseModel):
    symbol: str

@router.post("/analyze-symbol")
async def analyze_symbol_endpoint(req: SymbolReq, user=Depends(get_current_user)):
    return dummy_holding_response
    return await run_in_threadpool(get_linkup_symbol_analysis, req.symbol)

@router.post("/news-summary")
async def portfolio_news_summary(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    days_back: int = Query(7, ge=1, le=30),
    per_symbol_limit: int = Query(6, ge=1, le=20),
):
    holdings = get_all_holdings(user.id, db)
    symbols = [h.symbol for h in holdings]
    if not symbols:
        return {"summary": "", "highlights": [], "risks": [], "per_symbol": {}, "sentiment": 0, "sources": []}

    # 1) Fetch recent news per symbol
    news_by_symbol = await get_company_news_for_symbols(
        symbols, days_back=days_back, limit_per_symbol=per_symbol_limit
    )

    # 2) Summarize as a portfolio brief
    news_by_symbol_dict = {
        symbol: [item.__dict__ if hasattr(item, "__dict__") else dict(item) for item in items]
        for symbol, items in news_by_symbol.items()
    }
    summary_text = await summarize_portfolio_news(news_by_symbol_dict, symbols=symbols)

    urls: list[str] = []
    for arr in news_by_symbol.values():
        for it in arr:
            u = it.get("url")
            if u and u not in urls:
                urls.append(u)

    return {
        "summary": summary_text,
        "highlights": [],       # (optional) you can derive separately later
        "risks": [],            # (optional)
        "per_symbol": {},       # (optional)
        "sentiment": 0,         # (optional)
        "sources": urls[:8],
    }


dummy_holding_response = {
    "summary": "Apple Inc. (AAPL) stock has recently hit record highs in 2025, driven primarily by strong demand for the iPhone 17 series in key markets like the U.S. and China. Loop Capital upgraded the stock from hold to buy, raising the price target significantly to $315, citing a multi-year iPhone replacement cycle and ongoing shipment expansion through 2027. Other analysts, including Evercore ISI and Goldman Sachs, have also raised price targets and ratings, expecting upside in upcoming earnings and continued momentum. Apple's market capitalization is nearing $4 trillion, making it the second-most valuable company globally, surpassing Microsoft. Despite a challenging year with tariff impacts and underperformance relative to some tech peers, recent strong sales and product innovation have revitalized investor confidence. Risks include geopolitical tensions affecting supply chains and competitive pressures. Technically, the stock is in an uptrend with new all-time highs and strong momentum indicators. Key upcoming events include the fiscal Q4 earnings report expected on October 30, 2025. Overall, the outlook is bullish with significant upside potential based on product demand and strategic growth in services and AI integration.",
    "latest_developments": [
        {
            "headline": "Apple Share Price Hits First Record of 2025 on iPhone Optimism",
            "date": "2025-10-20T00:00:00Z",
            "source": "Bloomberg",
            "url": "https://www.bloomberg.com/news/articles/2025-10-20/apple-nears-record-high-as-loop-sees-25-upside-on-iphone-demand",
            "cause": "Loop Capital upgraded Apple stock to buy citing positive iPhone demand trends and a long-awaited replacement cycle starting.",
            "impact": "Stock hit record highs, up over 50% since April, turning positive for the year.",
            "assets_affected": [
                "AAPL"
            ]
        },
        {
            "headline": "Apple nears $4 trillion valuation as shares surge on strong iPhone 17 demand",
            "date": "2025-10-20T00:00:00Z",
            "source": "Reuters",
            "url": "https://www.reuters.com/world/asia-pacific/apple-closes-4-trillion-valuation-data-shows-strong-demand-iphone-17-2025-10-20/",
            "cause": "Strong momentum for iPhone 17 sales in US and China, with Evercore ISI adding Apple to Tactical Outperform list expecting earnings beat.",
            "impact": "Shares surged to all-time highs, market cap near $4 trillion.",
            "assets_affected": [
                "AAPL"
            ]
        },
        {
            "headline": "Apple stock rallies after strong iPhone 17 U.S., China sales",
            "date": "2025-10-20T00:00:00Z",
            "source": "CNBC",
            "url": "https://www.cnbc.com/2025/10/20/apple-stock-iphone-17-us-china-sales.html",
            "cause": "Strong sales of iPhone 17 in key markets and positive analyst notes from Loop Capital and others.",
            "impact": "Stock up about 5% in 2025, rallied 24% in last 3 months, with expectations of upside to September quarter results.",
            "assets_affected": [
                "AAPL"
            ]
        },
        {
            "headline": "Apple Tops Microsoft As World’s Second-Most Valuable Company After Stock Rally",
            "date": "2025-10-20T00:00:00Z",
            "source": "Forbes",
            "url": "https://www.forbes.com/sites/antoniopequenoiv/2025/10/20/apple-tops-microsoft-as-worlds-second-most-valuable-company-after-stock-rally/",
            "cause": "Stock rally fueled by heightened demand for iPhone 17 pushed market cap near $4 trillion.",
            "impact": "Apple overtook Microsoft as second-most valuable company globally.",
            "assets_affected": [
                "AAPL"
            ]
        },
        {
            "headline": "Jim Cramer says Apple's rally to a record shows why you should hold, not sell, the stock",
            "date": "2025-10-20T00:00:00Z",
            "source": "CNBC",
            "url": "https://www.cnbc.com/2025/10/20/cramer-apple-rally-stock.html",
            "cause": "Strong iPhone demand, services growth, and market momentum.",
            "impact": "Encourages investors to hold Apple stock due to long-term potential and momentum.",
            "assets_affected": [
                "AAPL"
            ]
        }
    ],
    "catalysts": [
        {
            "date": "2025-10-30",
            "type": "earnings",
            "description": "Apple's fiscal Q4 earnings report expected, with analysts anticipating upside due to strong iPhone 17 sales and services growth.",
            "expected_direction": "up",
            "magnitude_basis": "Depends on actual sales figures and guidance for December quarter.",
            "confidence": 0.8
        },
        {
            "date": "2025-10-20",
            "type": "product",
            "description": "Launch and strong sales momentum of iPhone 17 series, including potential new designs and AI features.",
            "expected_direction": "up",
            "magnitude_basis": "Based on early sales data and analyst upgrades.",
            "confidence": 0.9
        }
    ],
    "risks": [
        {
            "risk": "Geopolitical tensions and tariffs",
            "why_it_matters": "Apple's supply chain is heavily reliant on China; tariffs and political issues can impact costs and production.",
            "monitor": "Trade policies, government announcements, and supply chain reports."
        },
        {
            "risk": "Competition and market saturation",
            "why_it_matters": "Strong competition in smartphones and tech services could limit growth and margins.",
            "monitor": "Market share data, competitor product launches, and consumer trends."
        }
    ],
    "valuation": {
        "multiples": {
            "pe_ttm": 39.49,
            "fwd_pe": 35,
            "ps_ttm": 7.5,
            "ev_ebitda": 25
        },
        "peer_set": [
            "Microsoft",
            "Alphabet",
            "Meta",
            "Nvidia",
            "Amazon"
        ]
    },
    "technicals": {
        "trend": "Uptrend",
        "levels": {
            "support": 240,
            "resistance": 265
        },
        "momentum": {
            "rsi": 70,
            "comment": "Strong momentum with RSI near overbought levels, indicating bullish sentiment but watch for potential short-term pullbacks."
        }
    },
    "key_dates": [
        {
            "date": "2025-10-30",
            "event": "Fiscal Q4 earnings report"
        }
    ],
    "scenarios": {
        "bull": "Continued strong iPhone sales and services growth drive stock above $315 with market cap surpassing $4 trillion.",
        "base": "Steady growth with moderate upside as iPhone replacement cycle sustains demand and services expand, stock trading around $270-$300.",
        "bear": "Geopolitical issues or supply chain disruptions cause sales to falter, leading to stock correction below $240."
    },
    "disclaimer": "This analysis is based on publicly available information as of October 2025 and does not constitute financial advice. Investors should conduct their own research or consult a financial advisor before making investment decisions."
}