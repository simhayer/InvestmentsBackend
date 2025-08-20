
from sqlalchemy.orm import relationship
from database import Base
from sqlalchemy.orm import  Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

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

