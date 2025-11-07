# models/portfolio_analysis.py
from __future__ import annotations
from sqlalchemy import DateTime, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from database import Base
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

class PortfolioAnalysis(Base):
    __tablename__ = "portfolio_analyses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(nullable=False, index=True)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)  # full analysis payload
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_portfolio_analysis_user"),
    )
