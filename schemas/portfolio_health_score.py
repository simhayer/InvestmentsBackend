# types/portfolio_health_score_v2.py
from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

class HealthSubscores(BaseModel):
    diversification: int = Field(..., ge=0, le=60)
    risk_balance: int = Field(..., ge=0, le=40)
    sector_region: int = Field(..., ge=0, le=25)
    quality: int = Field(..., ge=0, le=15)

class PortfolioHealthScoreResponse(BaseModel):
    status: Literal["ok"] = "ok"
    user_id: str
    currency: str
    as_of: int

    score: int = Field(..., ge=0, le=100)
    grade: Literal["A", "B", "C", "D", "F"]
    baseline: Literal["balanced", "growth", "conservative"]

    subscores: HealthSubscores

    # exposures
    sector_weights_pct: Dict[str, float] = Field(default_factory=dict)
    region_weights_pct: Dict[str, float] = Field(default_factory=dict)

    insights: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
