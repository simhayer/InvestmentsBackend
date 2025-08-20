from pydantic import BaseModel
from typing import TypedDict

class HoldingInputPydantic(BaseModel):
    symbol: str
    name: str
    quantity: float
    purchase_price: float
    current_price: float
    type: str          # consider renaming to asset_type to avoid shadowing built-in
    institution: str
    currency: str

class HoldingInput(TypedDict):
    symbol: str
    name: str
    type: str
    quantity: float
    purchase_price: float
    current_price: float
    institution: str
    currency: str
