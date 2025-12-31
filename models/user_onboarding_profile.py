# models/user_onboarding.py
from sqlalchemy import String, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB  # if you're on Postgres
from datetime import datetime
# If not Postgres, use sqlalchemy.JSON instead:
# from sqlalchemy import JSON as JSONB

from database import Base

class UserOnboardingProfile(Base):
    __tablename__ = "user_onboarding_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    # Status / progress
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    current_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # High-signal fields
    time_horizon: Mapped[str | None] = mapped_column(String(16), nullable=True)  # short/medium/long
    primary_goal: Mapped[str | None] = mapped_column(String(32), nullable=True)  # growth/income/preserve/save_for_goal
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)    # low/medium/high
    experience_level: Mapped[str | None] = mapped_column(String(16), nullable=True)  # beginner/intermediate/advanced

    # Optional personalization
    age_band: Mapped[str | None] = mapped_column(String(16), nullable=True)  # 18_24, 25_34, ...
    country: Mapped[str | None] = mapped_column(String(2), nullable=True)    # CA, US, etc

    # Preferences as JSON blobs (flexible)
    asset_preferences: Mapped[dict | None] = mapped_column(JSONB, nullable=True)       # {"stocks":true,"etfs":true,"crypto":false}
    style_preference: Mapped[str | None] = mapped_column(String(24), nullable=True)    # set_and_forget/hands_on/news_driven
    notification_level: Mapped[str | None] = mapped_column(String(16), nullable=True)  # minimal/balanced/frequent
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="onboarding_profile")
