from pydantic import BaseModel
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    type: str
    name: Optional[str] = None
    currency: Optional[str] = "USD"

class HoldingUpdate(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None
    type: Optional[str] = None
    currency: Optional[str] = None
