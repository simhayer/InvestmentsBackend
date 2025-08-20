from database import Base
from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import  Mapped, mapped_column
from sqlalchemy import  DateTime, UniqueConstraint

class UserAccess(Base):
    __tablename__ = "user_access_tokens"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[int] = mapped_column(nullable=False)
    access_token: Mapped[str] = mapped_column(nullable=False)
    item_id: Mapped[str] = mapped_column(nullable=True)
    institution_id: Mapped[str] = mapped_column(nullable=True)
    institution_name: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("user_id", "institution_id", name="user_institution_unique"),
    )
