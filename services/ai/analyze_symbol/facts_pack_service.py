from typing import Dict, List, Any
from services.ai.analyze_symbol.facts_pack_schema import (
    ValuationFacts,
    GrowthFacts,
    ProfitabilityFacts,
    LeverageFacts,
    EventFacts,
    FactsPack,
)

def build_price_facts(
    quote: Dict[str, Any],
    market_snapshot: Dict[str, Any],
) -> dict:
    price = quote.get("c")
    change_pct = quote.get("dp", 0.0)

    ma_50 = market_snapshot.get("ma_50")
    ma_200 = market_snapshot.get("ma_200")

    vs_50 = "above" if ma_50 and price > ma_50 else "below"
    vs_200 = "above" if ma_200 and price > ma_200 else "below"

    if vs_50 == "above" and vs_200 == "above":
        regime = "strong_uptrend"
    elif vs_50 == "below" and vs_200 == "above":
        regime = "compression"
    elif vs_50 == "below" and vs_200 == "below":
        regime = "downtrend"
    else:
        regime = "range_bound"

    return {
        "last": price,
        "change_1d_pct": change_pct,
        "vs_50dma": vs_50,
        "vs_200dma": vs_200,
        "trend_regime": regime,
    }


def build_valuation_facts(peer_ready: Dict[str, any]) -> ValuationFacts:
    pe = peer_ready.get("key_stats", {}).get("pe_ttm", {})
    company = pe.get("company")
    median = pe.get("peer_median")

    if company is None or median is None:
        rel = "in_line_with_peers"
    elif company < median:
        rel = "cheaper_than_peers"
    elif company > median:
        rel = "more_expensive_than_peers"
    else:
        rel = "in_line_with_peers"

    return ValuationFacts(
        relative_position=rel,
        sensitivity="growth_driven",  # default; LLM will reason on impact
    )


def build_growth_facts(peer_ready: Dict[str, any]) -> GrowthFacts:
    g = peer_ready.get("key_stats", {}).get("revenue_growth_yoy", {})
    pct = g.get("company_percentile")

    if pct is None:
        rank = "average"
    elif pct >= 80:
        rank = "top_quintile"
    elif pct >= 60:
        rank = "above_average"
    elif pct >= 40:
        rank = "average"
    elif pct >= 20:
        rank = "below_average"
    else:
        rank = "bottom_quintile"

    return GrowthFacts(
        revenue_yoy=g.get("company"),
        peer_rank=rank,
        trend="decelerating" if rank in {"below_average", "bottom_quintile"} else "stable",
    )

def build_profitability_facts(peer_ready: Dict[str, any]) -> ProfitabilityFacts:
    om = peer_ready.get("key_stats", {}).get("operating_margin", {})
    company = om.get("company")
    median = om.get("peer_median")

    if company is None or median is None:
        pos = "in_line"
    elif company > median:
        pos = "above_median"
    else:
        pos = "below_median"

    return ProfitabilityFacts(
        operating_margin=company,
        peer_position=pos,
        trend="stable",
    )

def build_leverage_facts(peer_ready: Dict[str, any]) -> LeverageFacts:
    d2e = peer_ready.get("key_stats", {}).get("debt_to_equity", {})
    company = d2e.get("company")

    if company is None:
        flag = "moderate"
    elif company < 0.5:
        flag = "low"
    elif company < 1.0:
        flag = "moderate"
    else:
        flag = "elevated"

    return LeverageFacts(
        flag=flag,
        note="Equity may be distorted by buybacks; assess leverage via cash flow coverage.",
    )

def build_event_facts(earnings_small: List[Dict[str, any]]) -> EventFacts:
    next_date = earnings_small[0]["date"] if earnings_small else "Unknown"
    return EventFacts(
        next_earnings=next_date,
        importance="high",
    )


def build_news_flags(news_items: List[Dict[str, any]]) -> List[str]:
    flags = []
    for n in news_items:
        txt = (n.get("headline", "") + n.get("summary", "")).lower()
        if "regulat" in txt:
            flags.append("regulatory_pressure")
        if "china" in txt:
            flags.append("china_exposure")
    return list(set(flags))


def build_sec_flags(sec_risks: List[Dict[str, any]]) -> List[str]:
    flags = []
    for r in sec_risks:
        txt = (r.get("text", "")).lower()
        if "supply" in txt:
            flags.append("supply_chain_dependency")
        if "services" in txt:
            flags.append("services_growth_focus")
    return list(set(flags))


def build_facts_pack(
    market_snapshot,
    quote,
    peer_ready,
    earnings_small,
    news_items,
    sec_risks,
    data_quality_notes,
) -> FactsPack:

    return FactsPack(
        price=build_price_facts(quote, market_snapshot),
        valuation=build_valuation_facts(peer_ready),
        growth=build_growth_facts(peer_ready),
        profitability=build_profitability_facts(peer_ready),
        leverage=build_leverage_facts(peer_ready),
        events=build_event_facts(earnings_small),
        news_flags=build_news_flags(news_items),
        sec_flags=build_sec_flags(sec_risks),
        data_quality_notes=data_quality_notes,
    )

def build_current_performance(facts: dict) -> dict:
    price = facts["price"]

    bullets = []

    bullets.append(
        f"Shares last traded at {price['last']:.2f}, "
        f"{price['change_1d_pct']:+.2f}% on the day."
    )

    bullets.append(
        f"Price is {price['vs_50dma']} the 50-day average and "
        f"{price['vs_200dma']} the 200-day average."
    )

    regime_map = {
        "strong_uptrend": "Momentum is firmly positive across timeframes.",
        "weak_uptrend": "Uptrend remains intact but momentum is weakening.",
        "compression": "Short-term weakness contrasts with longer-term support.",
        "range_bound": "Price action remains range-bound with no clear trend.",
        "downtrend": "Momentum is negative across timeframes.",
    }

    bullets.append(regime_map.get(price["trend_regime"], "Trend signals are mixed."))

    return {"bullets": bullets}

def build_key_risks(facts: dict) -> list[str]:
    risks = []

    growth = facts["growth"]
    if growth["peer_rank"] in {"below_average", "bottom_quintile"}:
        risks.append("Revenue growth trails peers, limiting valuation expansion.")

    leverage = facts["leverage"]
    if leverage["flag"] == "elevated":
        risks.append("Leverage appears elevated, increasing sensitivity to cash flow pressure.")

    if "regulatory_pressure" in facts["news_flags"]:
        risks.append("Regulatory scrutiny could impact operating flexibility or margins.")

    if "china_exposure" in facts["news_flags"]:
        risks.append("Exposure to China-related demand or supply chains adds geopolitical risk.")

    if "supply_chain_dependency" in facts["sec_flags"]:
        risks.append("Supply chain concentration could disrupt production or delivery timelines.")

    if len(risks) <= 2:
        risks.append(
            "Overall risk profile is concentrated in a small number of identifiable factors."
        )
    return risks[:5]

def build_price_outlook(facts: dict, core: dict) -> dict:
    bullets = []

    valuation = facts["valuation"]["relative_position"]
    growth_rank = facts["growth"]["peer_rank"]
    regime = facts["price"]["trend_regime"]

    if valuation == "cheaper_than_peers":
        bullets.append(
            "Valuation appears supportive relative to peers, reducing downside from multiple compression."
        )
    elif valuation == "more_expensive_than_peers":
        bullets.append(
            "Premium valuation leaves shares more sensitive to earnings or guidance disappointment."
        )

    if growth_rank in {"below_average", "bottom_quintile"}:
        bullets.append(
            "Slower growth profile may cap near-term upside unless momentum improves."
        )

    if regime == "compression":
        bullets.append(
            "Price action suggests a consolidation phase ahead of a potential catalyst."
        )
    elif regime == "strong_uptrend":
        bullets.append(
            "Positive momentum supports continuation if fundamentals remain intact."
        )

    bullets.append(
        "Near-term direction is likely to be driven by upcoming company-specific catalysts."
    )

    return {"bullets": bullets[:5]}

def build_watch_list(facts: dict, core: dict) -> list[str]:
    watch = []

    # Earnings
    if facts["events"]["next_earnings"] != "Unknown":
        watch.append(f"Upcoming earnings on {facts['events']['next_earnings']}")

    # Growth
    watch.append("Revenue growth trajectory vs prior quarters")

    # Margins
    watch.append("Operating margin stability and cost discipline")

    # Risks from flags
    if "regulatory_pressure" in facts["news_flags"]:
        watch.append("Regulatory updates affecting platform or pricing")

    if "china_exposure" in facts["news_flags"]:
        watch.append("China demand trends and supply chain developments")

    return watch[:5]
