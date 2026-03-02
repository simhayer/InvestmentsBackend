from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_symbol(value: str) -> str:
    symbol = (value or "").strip().upper()
    if not symbol or len(symbol) > 20:
        raise ValueError("symbol must be 1-20 characters")
    return symbol


class WatchlistItemCreate(BaseModel):
    symbol: str
    note: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class WatchlistCreate(BaseModel):
    name: str
    is_default: bool = False
    symbols: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name or len(name) > 120:
            raise ValueError("name must be 1-120 characters")
        return name

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for symbol in value:
            normalized = _normalize_symbol(symbol)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out


class WatchlistUpdate(BaseModel):
    name: Optional[str] = None
    is_default: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        name = value.strip()
        if not name or len(name) > 120:
            raise ValueError("name must be 1-120 characters")
        return name


class WatchlistItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    note: Optional[str] = None
    created_at: datetime


class WatchlistOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_default: bool
    created_at: datetime
    updated_at: datetime
    items: list[WatchlistItemOut] = Field(default_factory=list)
