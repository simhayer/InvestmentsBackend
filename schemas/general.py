from pydantic import BaseModel, field_validator
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

ALLOWED_HOLDING_TYPES = {"stock", "etf", "cryptocurrency", "mutual_fund", "bond", "option"}

class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    type: str
    name: Optional[str] = None
    currency: Optional[str] = "USD"

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_nonempty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or len(v) > 20:
            raise ValueError("symbol must be 1-20 characters")
        return v

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @field_validator("purchase_price")
    @classmethod
    def price_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("purchase_price must be non-negative")
        return v

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALLOWED_HOLDING_TYPES:
            raise ValueError(f"type must be one of: {', '.join(sorted(ALLOWED_HOLDING_TYPES))}")
        return v

    @field_validator("currency")
    @classmethod
    def currency_valid(cls, v: Optional[str]) -> str:
        if v is None:
            return "USD"
        v = v.strip().upper()
        if len(v) != 3:
            raise ValueError("currency must be a 3-letter code (e.g. USD, CAD)")
        return v


class HoldingUpdate(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None
    type: Optional[str] = None
    currency: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_nonempty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().upper()
        if not v or len(v) > 20:
            raise ValueError("symbol must be 1-20 characters")
        return v

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @field_validator("purchase_price")
    @classmethod
    def price_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("purchase_price must be non-negative")
        return v

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ALLOWED_HOLDING_TYPES:
            raise ValueError(f"type must be one of: {', '.join(sorted(ALLOWED_HOLDING_TYPES))}")
        return v

    @field_validator("currency")
    @classmethod
    def currency_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().upper()
        if len(v) != 3:
            raise ValueError("currency must be a 3-letter code (e.g. USD, CAD)")
        return v
