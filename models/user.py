# models/user.py
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # NEW: link to Supabase auth.users.id (UUID string)
    supabase_user_id: Mapped[str] = mapped_column(unique=True, index=True)

    email: Mapped[str] = mapped_column(unique=True, index=True)

    # Make nullable because Supabase handles passwords now
    hashed_password: Mapped[str | None] = mapped_column(nullable=True)

    holdings = relationship("Holding", back_populates="owner")

    currency: Mapped[str] = mapped_column(default="USD")

    # how this currency was chosen
    # "default" = system default
    # "auto"    = inferred from holdings
    # "manual"  = user explicitly chose
    base_currency_source: Mapped[str] = mapped_column(
        String(16),
        default="default",
        nullable=False,
    )

    onboarding_profile = relationship(
        "UserOnboardingProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    subscription = relationship("UserSubscription", back_populates="user", uselist=False)
    watchlists = relationship("Watchlist", back_populates="owner", cascade="all, delete-orphan")
