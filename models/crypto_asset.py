# models/crypto_asset.py
from database import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, DateTime, func, Index, UniqueConstraint

class CryptoAsset(Base):
    __tablename__ = "crypto_assets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # e.g. "BTC", "ETH"
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)

    # optional nicer display name (you can fill later)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # binance as primary provider for now
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="binance")

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("provider", "symbol", name="uq_crypto_assets_provider_symbol"),
        Index("ix_crypto_assets_symbol", "symbol"),
    )
