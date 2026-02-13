from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, func, Boolean, false as sa_false
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # app-facing plan you gate on
    plan: Mapped[str] = mapped_column(String(16), default="free", nullable=False)   # free | premium | pro
    # stripe-facing status
    status: Mapped[str] = mapped_column(String(32), default="free", nullable=False)  # trialing | active | past_due | canceled | ...

    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false(), nullable=False)
    cancel_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="subscription")
