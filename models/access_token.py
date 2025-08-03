from sqlalchemy import Column, String, DateTime, UniqueConstraint
from database import Base
from datetime import datetime, timezone
import uuid

class UserAccess(Base):
    __tablename__ = "user_access_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    access_token = Column(String, nullable=False)
    item_id = Column(String, nullable=True)
    institution_id = Column(String, nullable=True)
    institution_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("user_id", "institution_id", name="user_institution_unique"),
    )
