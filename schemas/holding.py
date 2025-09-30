from typing import TypedDict

class HoldingInput(TypedDict):
    symbol: str
    name: str
    type: str
    quantity: float
    purchase_price: float
    current_price: float
    institution: str
    currency: str
