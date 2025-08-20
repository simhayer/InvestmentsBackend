from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base
# from .holding import Holding

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column()
    holdings = relationship("Holding", back_populates="owner")