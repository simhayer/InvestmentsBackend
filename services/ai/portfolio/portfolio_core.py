# services/ai/portfolio/portfolio_core.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PortfolioKeyInsight(BaseModel):
    insight: str
    evidence: str
    implication: str


class PortfolioThesisPoint(BaseModel):
    claim: str
    why_it_matters: str
    what_would_change_my_mind: str


class RebalanceIdea(BaseModel):
    idea: str
    why: str
    how: str
    tradeoff: str


class PortfolioRisk(BaseModel):
    risk: str
    why_it_matters: str
    what_to_watch: str


class PortfolioCoreAnalysis(BaseModel):
    # keep it strict to reduce fluff
    key_insights: List[PortfolioKeyInsight] = Field(..., min_length=4, max_length=4)
    portfolio_thesis: List[PortfolioThesisPoint] = Field(..., min_length=3, max_length=3)
    portfolio_risks: List[PortfolioRisk] = Field(..., min_length=3, max_length=3)
    rebalance_ideas: List[RebalanceIdea] = Field(..., min_length=3, max_length=3)
    what_to_watch_next: List[str] = Field(..., min_length=5, max_length=5)
