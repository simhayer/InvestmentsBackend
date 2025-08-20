from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from services.auth_service import get_current_user
from services.ai_service import analyze_investment_holding
from schemas.holding import HoldingInputPydantic
from utils.converters import to_holding_dict

router = APIRouter()

@router.post("/analyze-holding")
async def analyze_holding(
    holding: HoldingInputPydantic,
    user=Depends(get_current_user),
):
    # analyze_investment_holding is a *sync* function
    analysis = await run_in_threadpool(analyze_investment_holding, to_holding_dict(holding))
    return {"insight": analysis}
