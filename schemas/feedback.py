from typing import Literal

from pydantic import BaseModel, field_validator


FeedbackCategory = Literal["bug", "idea", "other"]


class FeedbackCreate(BaseModel):
    category: FeedbackCategory
    message: str
    email: str | None = None
    page_url: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        value = v.strip()
        if len(value) < 3:
            raise ValueError("message must be at least 3 characters")
        if len(value) > 5000:
            raise ValueError("message must be at most 5000 characters")
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        if len(value) > 320:
            raise ValueError("email must be at most 320 characters")
        return value

    @field_validator("page_url")
    @classmethod
    def validate_page_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        if len(value) > 2048:
            raise ValueError("page_url must be at most 2048 characters")
        return value
