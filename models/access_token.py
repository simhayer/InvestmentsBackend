from sqlalchemy import Column, String
import uuid
from database import Base

class UserAccess(Base):
    __tablename__ = "user_access_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, unique=True, nullable=False)
    access_token = Column(String, nullable=False)
    item_id = Column(String, nullable=True)
