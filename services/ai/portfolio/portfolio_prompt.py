# services/ai/portfolio/portfolio_prompt.py
import json
from services.openai.client import llm
from services.ai.portfolio.portfolio_core import PortfolioCoreAnalysis


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


async def run_portfolio_core_llm(facts_pack: dict) -> dict:
    llm_core = llm.with_structured_output(PortfolioCoreAnalysis).bind(temperature=0.2)

    prompt = f"""
You are given a PORTFOLIO FACTS PACK. Treat it as correct.

Your job:
- Explain what drives this portfolio (focus on CORE holdings)
- Call out concentration and hidden risk (focus on RISK AMPLIFIERS)
- Keep it grounded in numbers from the pack (no generic advice)

OUTPUT RULES (STRICT)
- key_insights: EXACTLY 4
- portfolio_thesis: EXACTLY 3
- portfolio_risks: EXACTLY 3
- rebalance_ideas: EXACTLY 3 (each must include a clear tradeoff)
- what_to_watch_next: EXACTLY 5 short bullets

QUALITY RULES
- Every key_insight must reference at least ONE metric from the pack:
  (top_5_weight_pct, core_weight_share_pct, risk_weight_share_pct, HHI, day_pl, unrealized_pl_pct, a holding weight, etc.)
- Evidence must be a short plain-English sentence with numbers.
- Do NOT mention “I am not a financial advisor”. Just be neutral and practical.
- Do NOT recommend specific tickers to buy/sell. You can suggest direction (reduce concentration, add diversification, etc.)

PORTFOLIO FACTS PACK:
{json_dumps(facts_pack)}
""".strip()

    core = await llm_core.ainvoke(prompt)
    return core.model_dump()
