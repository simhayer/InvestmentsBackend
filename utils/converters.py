# utils/converters.py
from typing import cast
from pydantic import BaseModel
from schemas.holding import HoldingInput

def to_holding_dict(m: BaseModel) -> HoldingInput:
    """
    Convert a Pydantic model to HoldingDict. Keeps keys consistent with service layer.
    """
    d = m.model_dump(by_alias=False, exclude_none=True)  # keys like asset_type, not "type"
    return cast(HoldingInput, d)
