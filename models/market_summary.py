# app/models/market.py
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import JSON, DateTime, String, func
from database import Base

class MarketSummary(Base):
    __tablename__ = "market_summaries"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of: Mapped[DateTime] = mapped_column(DateTime(timezone=True), index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    payload: Mapped[dict] = mapped_column(JSON)  # your whole response.data
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
