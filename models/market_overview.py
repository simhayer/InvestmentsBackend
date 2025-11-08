from sqlalchemy import Integer, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from database import Base
from sqlalchemy.orm import  mapped_column

class MarketOverviewLatest(Base):
    __tablename__ = "market_overview_latest"

    # id = Column(Integer, primary_key=True, index=True)
    id = mapped_column(Integer, primary_key=True, index=True)
    symbols = mapped_column(ARRAY(Text), nullable=False)
    items = mapped_column(JSONB, nullable=False)
    ai_summary = mapped_column(Text, nullable=True)
    fetched_at = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class MarketOverviewHistory(Base):
    __tablename__ = "market_overview_history"

    ts = mapped_column(DateTime(timezone=True), primary_key=True, server_default=func.now())
    symbols = mapped_column(ARRAY(Text), nullable=False)
    items = mapped_column(JSONB, nullable=False)
    ai_summary = mapped_column(Text, nullable=True)
