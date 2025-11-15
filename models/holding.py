
from sqlalchemy.orm import relationship
from database import Base
from sqlalchemy.orm import  Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from pydantic import BaseModel, ConfigDict

class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    type: Mapped[str] = mapped_column()
    quantity: Mapped[float] = mapped_column()
    purchase_price: Mapped[float] = mapped_column()
    current_price: Mapped[float] = mapped_column()
    value: Mapped[float] = mapped_column()
    currency: Mapped[str] = mapped_column()
    institution: Mapped[str] = mapped_column()
    account_name: Mapped[str] = mapped_column()
    source: Mapped[str] = mapped_column()
    external_id: Mapped[str] = mapped_column()

    # NEW fields
    purchase_amount_total: Mapped[float | None] = mapped_column(nullable=True)
    purchase_unit_price: Mapped[float | None] = mapped_column(nullable=True)
    unrealized_pl: Mapped[float | None] = mapped_column(nullable=True)
    unrealized_pl_pct: Mapped[float | None] = mapped_column(nullable=True)
    current_value: Mapped[float | None] = mapped_column(nullable=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    owner = relationship("User", back_populates="holdings")

class HoldingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # DB fields (names match the DTO you want to send)
    id: int
    user_id: int
    symbol: str
    name: str | None = None
    type: str | None = None             # expose as "type" for frontend naming
    quantity: float | None = None
    purchase_price: float | None = None
    current_price: float | None = None
    value: float | None = None
    currency: str | None = None
    institution: str | None = None
    account_name: str | None = None
    source: str | None = None
    external_id: str | None = None

    # Transient/computed fields (NOT in DB)
    previous_close: float | None = None
    price_status: str | None = None
    previous_close: float | None = None
    day_pl: float | None = None
    unrealized_pl: float | None = None
    weight: float | None = None         # percentage of portfolio, if computed

    # NEW fields
    purchase_amount_total: float | None = None   # TOTAL cost basis
    purchase_unit_price: float | None = None     # per-unit cost
    unrealized_pl: float | None = None           # total P/L
    unrealized_pl_pct: float | None = None       # % P/L
    current_value: float | None = None           

    # If you want, add derived fields via @computed_field in Pydantic v2

def to_dto(h: Holding) -> HoldingOut:
    # map ORM asset_type -> DTO type
    dto = HoldingOut.model_validate(h)
    dto.type = h.type
    return dto