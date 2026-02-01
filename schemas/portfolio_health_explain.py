# types/portfolio_health_explain.py
from typing import List
from pydantic import BaseModel, Field

class PortfolioHealthExplainRequest(BaseModel):
    health_score: dict  # pass the full health-score-v2 response


class PortfolioHealthExplainResponse(BaseModel):
    summary: str
    key_drivers: List[str] = Field(default_factory=list)
    what_helped: List[str] = Field(default_factory=list)
    what_hurt: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
