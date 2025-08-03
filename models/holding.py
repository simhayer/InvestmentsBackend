from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String)              # Ticker (e.g. BTC, AAPL)
    name = Column(String)                # Full name of the security
    type = Column(String)                # "equity", "crypto", etc.
    quantity = Column(Float)             # How much the user owns
    purchase_price = Column(Float)          # Price at which the user bought the asset
    current_price = Column(Float)        # Latest market price
    value = Column(Float)                # Quantity × current_price
    currency = Column(String)            # "USD", etc.
    institution = Column(String)         # "Wealthsimple", etc.
    account_name = Column(String)        # e.g. "401k", "IRA"
    source = Column(String)              # "plaid" or "manual"
    external_id = Column(String)
    
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="holdings")
