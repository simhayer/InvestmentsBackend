# models/user.py
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
