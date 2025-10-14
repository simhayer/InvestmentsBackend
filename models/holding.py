
from sqlalchemy.orm import relationship
from database import Base
from sqlalchemy.orm import  Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from pydantic import BaseModel, ConfigDict

class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column()              # Ticker (e.g. BTC, AAPL)
    name: Mapped[str] = mapped_column()                # Full name of the security
    type: Mapped[str] = mapped_column()                # "equity", "crypto", etc.
    quantity: Mapped[float] = mapped_column()          # How much the user owns
    purchase_price: Mapped[float] = mapped_column()    # Price at which the user bought the asset
    current_price: Mapped[float] = mapped_column()     # Latest market price
    value: Mapped[float] = mapped_column()             # Quantity × current_price
    currency: Mapped[str] = mapped_column()            # "USD", etc.
    institution: Mapped[str] = mapped_column()         # "Wealthsimple", etc.
    account_name: Mapped[str] = mapped_column()        # e.g. "401k", "IRA"
    source: Mapped[str] = mapped_column()              # "plaid" or "manual"
    external_id: Mapped[str] = mapped_column()

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

    # If you want, add derived fields via @computed_field in Pydantic v2

def to_dto(h: Holding) -> HoldingOut:
    # map ORM asset_type -> DTO type
    dto = HoldingOut.model_validate(h)
    dto.type = h.type
    return dto