from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    type: str
