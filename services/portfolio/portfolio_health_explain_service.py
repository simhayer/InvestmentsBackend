# services/portfolio/portfolio_health_explain_service.py
from __future__ import annotations

import json

from services.openai.client import llm
from schemas.portfolio_health_explain import (
    PortfolioHealthExplainRequest,
    PortfolioHealthExplainResponse,
)

SYSTEM_PROMPT = """
You are a professional investment analyst.

Your task is to EXPLAIN a portfolio health score that was already calculated.
You must NOT:
- recalculate scores
- introduce new data
- give financial advice

You MUST:
- explain WHY the score looks the way it does
- reference the provided metrics and insights only
- keep tone calm, factual, and professional
- write in plain English (no jargon)

Output MUST be valid JSON matching the provided schema.
"""

USER_PROMPT_TEMPLATE = """
Portfolio Health Score Data:
{payload}

Explain this score clearly for an investor.
Focus on:
- biggest drivers
- what helped vs what hurt
- practical next steps (non-advisory)
"""


async def explain_portfolio_health(
    req: PortfolioHealthExplainRequest,
) -> PortfolioHealthExplainResponse:
    payload = json.dumps(req.health_score, indent=2)

    prompt = USER_PROMPT_TEMPLATE.format(payload=payload)

    result = await llm.with_structured_output(
        PortfolioHealthExplainResponse
    ).ainvoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return result
