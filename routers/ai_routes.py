from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from services.helpers.linkup.single_stock_analysis_agent import call_link_up_for_single_stock
from services.holding_service import get_all_holdings
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from fastapi import APIRouter, Query
from models.user import User
from services.portfolio_service import get_or_compute_portfolio_analysis
from services.supabase_auth import get_current_db_user
from services.finnhub_service import FinnhubService
from routers.finnhub_routes import get_finnhub_service

router = APIRouter()

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    force: bool = Query(False, description="Bypass cache and recompute now"),
    user: User = Depends(get_current_db_user),
    db: Session = Depends(get_db),
    finnhub: FinnhubService = Depends(get_finnhub_service),
):
    data, meta = await get_or_compute_portfolio_analysis(
        user_id=str(user.id),
        db=db,
        base_currency=user.currency,  # "USD" or "CAD"
        days_of_news=7,
        targets={"Equities": 60, "Bonds": 30, "Cash": 10},
        force=force,
        finnhub=finnhub,
    )
    if data is None:
        reason = meta.get("reason") if isinstance(meta, dict) else "unknown"
        raise HTTPException(status_code=400, detail=f"Cannot analyze portfolio: {reason}")

    return {
        "status": "ok",
        "user_id": user.id,
        **meta,
        "ai_layers": data["ai_layers"],
    }

class SymbolReq(BaseModel):
    symbol: str

@router.post("/analyze-symbol")
async def analyze_symbol_endpoint(req: SymbolReq, user=Depends(get_current_db_user)):
    # return dummy_holding_response
    return await run_in_threadpool(call_link_up_for_single_stock, req.symbol)

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

dummy_portfolio_response = {
    "status": "ok",
    "user_id": 2,
    "ai_layers": {
        "data": {
            "latest_developments": [
                {
                    "headline": "Bitcoin price volatility and institutional accumulation continue in late October 2025",
                    "date": "2025-10-28",
                    "source": "Coindesk, Bitget, CoinGecko, Yahoo Finance",
                    "cause": "Market correction after October 10-11 crash, Fed rate cut expectations, institutional buying",
                    "impact": "Bitcoin price fluctuated between $104,000 and $126,000, with institutional investors accumulating BTC; market shows cautious sentiment with potential for further volatility",
                    "assets_affected": [
                        "BTC"
                    ]
                },
                {
                    "headline": "iShares MSCI Brazil ETF (EWZ) hits new 52-week high",
                    "date": "2025-11-03",
                    "source": "Daily Political",
                    "cause": "Positive market sentiment and institutional buying",
                    "impact": "EWZ reached a new 52-week high at $31.46, indicating strong performance in Brazilian equities",
                    "assets_affected": [
                        "EWZ"
                    ]
                },
                {
                    "headline": "Southside Bancshares Inc. (SBSI) sees increased institutional holdings and strategic review",
                    "date": "2025-10-31",
                    "source": "Daily Political, QuotedData",
                    "cause": "Institutional buying and ongoing strategic review by management",
                    "impact": "State of New Jersey Common Pension Fund D increased holdings by 4%; SBSI stock price around $28.37 with analyst price targets between $31 and $36",
                    "assets_affected": [
                        "SBSI"
                    ]
                },
                {
                    "headline": "Alberni Clayoquot Health Network (ACHN) continues community health initiatives",
                    "date": "2025-11-04",
                    "source": "ACHN official site",
                    "cause": "Ongoing regional health network activities",
                    "impact": "ACHN supports community health through networking and capacity building in the Alberni Clayoquot region",
                    "assets_affected": [
                        "ACHN"
                    ]
                },
                {
                    "headline": "ewz (Elektrizitätswerk der Stadt Zürich) advances renewable energy projects and local electricity communities",
                    "date": "2025-09 to 2025-11",
                    "source": "ewz official site and newsroom",
                    "cause": "Municipal initiatives and regulatory changes effective 2026",
                    "impact": "ewz promotes solar power production, local electricity communities, and sustainable energy solutions in Zurich",
                    "assets_affected": [
                        "EWZ"
                    ]
                }
            ],
            "catalysts": [
                {
                    "date": "2025-10-28",
                    "type": "macro",
                    "description": "Federal Open Market Committee (FOMC) expected to cut benchmark rates by 25 basis points to 4.00%-4.25% range",
                    "expected_direction": "up",
                    "magnitude_basis": "Rate cut generally supportive of risk assets including Bitcoin",
                    "confidence": 0.7,
                    "assets_affected": [
                        "BTC"
                    ]
                },
                {
                    "date": "2025-10-28",
                    "type": "earnings",
                    "description": "MicroStrategy continues Bitcoin accumulation with 390 BTC purchase in late October",
                    "expected_direction": "up",
                    "magnitude_basis": "Institutional accumulation signals confidence and potential price support",
                    "confidence": 0.8,
                    "assets_affected": [
                        "BTC"
                    ]
                },
                {
                    "date": "2025-11-17",
                    "type": "vote",
                    "description": "Schroder BSC Social Impact (SBSI) to update shareholders on strategic review at annual general meeting",
                    "expected_direction": "unclear",
                    "magnitude_basis": "Outcome of strategic review could impact share price and direction",
                    "confidence": 0.6,
                    "assets_affected": [
                        "SBSI"
                    ]
                },
                {
                    "date": "2025-11-10 to 2025-11-21",
                    "type": "macro",
                    "description": "COP30 Climate Summit in Brazil (Belem)",
                    "expected_direction": "unclear",
                    "magnitude_basis": "Global climate policy developments may influence renewable energy sectors and related equities",
                    "confidence": 0.5,
                    "assets_affected": [
                        "EWZ"
                    ]
                }
            ],
            "scenarios": {
                "bull": "Bitcoin price recovers above $120,000 supported by continued institutional buying and favorable Fed policy; EWZ continues upward trend with strong Brazilian economic data; SBSI strategic review leads to positive restructuring and share price appreciation; ACHN expands community health initiatives successfully; ewz benefits from renewable energy policies and local electricity community growth.",
                "base": "Bitcoin trades sideways in $105,000-$115,000 range with moderate volatility; EWZ maintains current levels with some fluctuations; SBSI remains under strategic review with no major changes; ACHN and ewz continue steady operations without significant new developments.",
                "bear": "Bitcoin falls below $100,000 due to macroeconomic uncertainties and profit-taking; EWZ declines on emerging market risks; SBSI faces negative outcomes from strategic review leading to share price pressure; ACHN faces funding or operational challenges; ewz impacted by regulatory or market setbacks in energy sector.",
                "probabilities": {
                    "bull": 0.4,
                    "base": 0.4,
                    "bear": 0.2
                }
            },
            "actions": [
                {
                    "title": "Monitor Bitcoin price and institutional accumulation trends",
                    "rationale": "Institutional buying and Fed policy are key drivers for BTC price direction; monitoring helps timely portfolio adjustments",
                    "impact": "medium",
                    "urgency": "high",
                    "effort": "low",
                    "targets": [
                        "BTC"
                    ],
                    "category": "alert"
                },
                {
                    "title": "Review SBSI strategic review outcomes post-AGM on 17 December 2025",
                    "rationale": "Strategic decisions could materially affect SBSI valuation and risk profile",
                    "impact": "high",
                    "urgency": "medium",
                    "effort": "medium",
                    "targets": [
                        "SBSI"
                    ],
                    "category": "research"
                },
                {
                    "title": "Assess EWZ exposure in renewable energy and local electricity community initiatives",
                    "rationale": "Renewable energy policies and local community projects may enhance EWZ growth prospects",
                    "impact": "medium",
                    "urgency": "medium",
                    "effort": "medium",
                    "targets": [
                        "EWZ"
                    ],
                    "category": "research"
                },
                {
                    "title": "Evaluate ACHN's community health initiatives for potential partnership or impact",
                    "rationale": "ACHN's regional health network activities may influence local health outcomes and related investments",
                    "impact": "low",
                    "urgency": "low",
                    "effort": "low",
                    "targets": [
                        "ACHN"
                    ],
                    "category": "research"
                },
                {
                    "title": "Consider macroeconomic developments around FOMC meetings and COP30 summit",
                    "rationale": "Global economic and climate policy events can impact portfolio assets and risk environment",
                    "impact": "medium",
                    "urgency": "medium",
                    "effort": "low",
                    "targets": [
                        "BTC",
                        "EWZ"
                    ],
                    "category": "alert"
                }
            ],
            "risks_list": [
                {
                    "risk": "Bitcoin price volatility and regulatory uncertainty",
                    "why_it_matters": "BTC price is sensitive to macroeconomic policy, market sentiment, and regulatory changes, affecting portfolio value",
                    "monitor": "BTC price levels, Fed announcements, regulatory news",
                    "assets_affected": [
                        "BTC"
                    ]
                },
                {
                    "risk": "Strategic review outcomes for SBSI",
                    "why_it_matters": "SBSI's ongoing strategic review may lead to restructuring or changes impacting share price and liquidity",
                    "monitor": "SBSI AGM announcements and management communications",
                    "assets_affected": [
                        "SBSI"
                    ]
                },
                {
                    "risk": "Emerging market and geopolitical risks affecting EWZ and EWZ holdings",
                    "why_it_matters": "EWZ exposure to renewable energy and local markets may be influenced by policy changes and geopolitical events",
                    "monitor": "EWZ news, regulatory changes, geopolitical developments in Brazil and Switzerland",
                    "assets_affected": [
                        "EWZ"
                    ]
                },
                {
                    "risk": "Operational and funding risks for ACHN",
                    "why_it_matters": "ACHN's community health initiatives depend on funding and regional cooperation, which may affect performance",
                    "monitor": "ACHN reports and regional health policy updates",
                    "assets_affected": [
                        "ACHN"
                    ]
                }
            ],
            "summary": "Recent developments show Bitcoin experiencing volatility but supported by institutional accumulation and expected Fed rate cuts, suggesting cautious optimism. EWZ reached new highs driven by positive market sentiment in Brazilian equities and renewable energy initiatives. SBSI is under strategic review with potential impact on its valuation. ACHN continues its community health efforts in its region. Key upcoming events include SBSI's AGM and the COP30 climate summit, which may influence portfolio assets. Monitoring macroeconomic policies, institutional activities, and strategic outcomes is recommended to manage risks and opportunities.",
            "disclaimer": "This information is provided for informational purposes only and does not constitute investment advice or recommendations. Investment decisions should be made based on individual circumstances and professional consultation."
        }
    }
}

dummy_holding_response_v2 = {
    "status": "ok",
    "user_id": 2,
    "ai_layers": {
        "latest_developments": [
            {
                "headline": "iShares MSCI Brazil ETF (EWZ) sets new 1-year high",
                "date": "2025-11-03",
                "source": "Daily Political",
                "url": "https://www.dailypolitical.com/2025/11/03/ishares-msci-brazil-etf-nysearcaewz-sets-new-1-year-high-should-you-buy.html",
                "cause": "Strong buying interest and institutional activity",
                "impact": "Shares reached a new 52-week high at $31.46, indicating bullish momentum",
                "assets_affected": [
                    "EWZ"
                ]
            },
            {
                "headline": "Institutional investors adjust holdings in iShares MSCI Brazil ETF (EWZ)",
                "date": "2025-10-29",
                "source": "Defense World",
                "url": "https://www.defenseworld.net/2025/10/29/brookstone-capital-management-sells-940-shares-of-ishares-msci-brazil-etf-ewz.html",
                "cause": "Portfolio rebalancing by institutional investors",
                "impact": "Brookstone Capital Management reduced holdings by 10.4%, while Sumitomo Mitsui Trust Group increased holdings by 152.9%",
                "assets_affected": [
                    "EWZ"
                ]
            },
            {
                "headline": "Southside Bancshares (SBSI) upgraded to Hold by Wall Street Zen",
                "date": "2025-11-02",
                "source": "Defense World",
                "url": "https://www.defenseworld.net/2025/11/02/southside-bancshares-nysesbsi-raised-to-hold-at-wall-street-zen.html",
                "cause": "Positive earnings surprise and valuation reassessment",
                "impact": "Rating upgrade from Sell to Hold, indicating stabilization",
                "assets_affected": [
                    "SBSI"
                ]
            }
        ],
        "catalysts": [
            {
                "date": "2025-11-02",
                "type": "earnings",
                "description": "Southside Bancshares reported quarterly earnings beating consensus estimates with EPS of $0.80 vs $0.72 expected",
                "expected_direction": "up",
                "magnitude_basis": "Earnings beat and positive analyst rating upgrade",
                "confidence": 0.7,
                "assets_affected": [
                    "SBSI"
                ]
            },
            {
                "date": "2025-11-03",
                "type": "market",
                "description": "iShares MSCI Brazil ETF (EWZ) reached a new 52-week high driven by institutional buying and positive market sentiment",
                "expected_direction": "up",
                "magnitude_basis": "Price breakout to new high with volume",
                "confidence": 0.75,
                "assets_affected": [
                    "EWZ"
                ]
            },
            {
                "date": "2025-10-30",
                "type": "macro",
                "description": "Federal Reserve interest rate cut of 25 basis points with indication of no further cuts in 2025 impacting Bitcoin price",
                "expected_direction": "unclear",
                "magnitude_basis": "Mixed market reaction with short-term price volatility",
                "confidence": 0.6,
                "assets_affected": [
                    "BTC"
                ]
            }
        ],
        "scenarios": {
            "bull": "Continued institutional accumulation in EWZ and positive earnings momentum in SBSI drive portfolio gains. Bitcoin recovers from recent volatility supported by macroeconomic easing and positive market sentiment.",
            "base": "EWZ maintains current levels with moderate volatility; SBSI stabilizes with steady earnings; Bitcoin experiences short-term fluctuations but remains range-bound.",
            "bear": "EWZ faces selling pressure from profit-taking; SBSI earnings disappoint leading to downgrade; Bitcoin declines further due to hawkish Fed stance and macro uncertainty.",
            "probabilities": {
                "bull": 0.4,
                "base": 0.45,
                "bear": 0.15
            }
        },
        "actions": [
            {
                "title": "Monitor EWZ institutional activity",
                "rationale": "Institutional buying and selling in EWZ is significant and may signal trend changes; monitoring can inform timely rebalancing.",
                "impact": "medium",
                "urgency": "medium",
                "effort": "low",
                "targets": [
                    "EWZ"
                ],
                "category": "research"
            },
            {
                "title": "Review SBSI earnings and analyst updates",
                "rationale": "Recent earnings beat and rating upgrade suggest potential stabilization; further updates may affect position sizing.",
                "impact": "medium",
                "urgency": "medium",
                "effort": "low",
                "targets": [
                    "SBSI"
                ],
                "category": "research"
            },
            {
                "title": "Assess Bitcoin exposure amid macro uncertainty",
                "rationale": "Fed rate cut and mixed signals create volatility; consider hedging or adjusting exposure based on risk tolerance.",
                "impact": "high",
                "urgency": "high",
                "effort": "medium",
                "targets": [
                    "BTC"
                ],
                "category": "hedge"
            },
            {
                "title": "Prepare for potential volatility around Fed announcements",
                "rationale": "Upcoming macro events may cause short-term market swings affecting BTC and equities; readiness can mitigate risks.",
                "impact": "medium",
                "urgency": "high",
                "effort": "medium",
                "targets": [
                    "BTC",
                    "EWZ"
                ],
                "category": "alert"
            }
        ],
        "risks_list": [
            {
                "risk": "Macroeconomic uncertainty impacting Bitcoin price",
                "why_it_matters": "Fed policy changes and global economic conditions can cause high volatility in BTC, affecting portfolio value.",
                "monitor": "Federal Reserve announcements, macroeconomic data releases",
                "assets_affected": [
                    "BTC"
                ]
            },
            {
                "risk": "Institutional rebalancing in emerging market equities (EWZ)",
                "why_it_matters": "Large institutional trades can cause price swings and affect liquidity in EWZ.",
                "monitor": "Institutional filings, trading volumes, price movements",
                "assets_affected": [
                    "EWZ"
                ]
            },
            {
                "risk": "Earnings volatility in regional banks (SBSI)",
                "why_it_matters": "Earnings surprises or downgrades can impact SBSI stock price and portfolio performance.",
                "monitor": "Quarterly earnings reports, analyst ratings",
                "assets_affected": [
                    "SBSI"
                ]
            }
        ],
        "performance_analysis": {
            "summary": "EWZ shows strength with new 52-week highs driven by institutional activity. SBSI has stabilized with recent earnings beat and rating upgrade. BTC experienced volatility due to Fed rate cut signals but maintains key support levels.",
            "leaders": [
                "EWZ"
            ],
            "laggards": [
                "BTC"
            ],
            "notable_shifts": [
                "SBSI upgraded from sell to hold"
            ]
        },
        "sentiment": {
            "overall_sentiment": "neutral",
            "sources_considered": [
                "Daily Political",
                "Defense World",
                "Bloomberg",
                "CoinDesk",
                "MarketBeat"
            ],
            "drivers": [
                {
                    "theme": "Institutional activity in EWZ",
                    "tone": "positive",
                    "impact": "Supports upward price momentum"
                },
                {
                    "theme": "Fed rate cut and macro uncertainty",
                    "tone": "negative",
                    "impact": "Creates volatility and cautious sentiment in BTC"
                },
                {
                    "theme": "Earnings beat and rating upgrade in SBSI",
                    "tone": "positive",
                    "impact": "Improves confidence in regional bank equity"
                }
            ],
            "summary": "Sentiment is mixed with positive developments in EWZ and SBSI offset by cautiousness in Bitcoin due to macroeconomic uncertainty."
        },
        "predictions": {
            "forecast_window": "30D",
            "assets": [
                {
                    "symbol": "EWZ",
                    "expected_direction": "up",
                    "expected_change_pct": 3,
                    "confidence": 0.7,
                    "rationale": "New 52-week highs and strong institutional buying suggest continued upward momentum in the near term."
                },
                {
                    "symbol": "SBSI",
                    "expected_direction": "neutral",
                    "expected_change_pct": 1,
                    "confidence": 0.6,
                    "rationale": "Recent earnings beat and rating upgrade support stability, but limited catalysts for strong moves."
                },
                {
                    "symbol": "BTC",
                    "expected_direction": "neutral",
                    "expected_change_pct": -2,
                    "confidence": 0.5,
                    "rationale": "Fed signals and macro uncertainty likely to keep Bitcoin volatile and range-bound in short term."
                }
            ]
        },
        "explainability": {
            "assumptions": [
                "Institutional trading activity influences EWZ price trends.",
                "Fed monetary policy impacts Bitcoin price volatility.",
                "Earnings reports affect SBSI stock performance."
            ],
            "confidence_overall": 0.65,
            "section_confidence": {
                "news": 0.7,
                "catalysts": 0.65,
                "actions": 0.6,
                "sentiment": 0.6,
                "predictions": 0.6,
                "scenarios": 0.6
            },
            "limitations": [
                "Limited direct news on ACHN and NHX105509 within the window.",
                "Market conditions can change rapidly beyond current data.",
                "Predictions are probabilistic and not guaranteed."
            ]
        },
        "section_confidence": {
            "scenarios": 0.6,
            "news": 0.7,
            "actions": 0.6
        },
        "summary": "Recent developments show positive momentum in EWZ and SBSI supported by institutional activity and earnings beats, while Bitcoin faces volatility due to macroeconomic factors. The portfolio should monitor these signals closely and prepare for potential short-term fluctuations.",
        "disclaimer": "This information is for informational purposes only and does not constitute investment advice or recommendations."
    }
}