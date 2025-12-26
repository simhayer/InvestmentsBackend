from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str

class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    type: str
