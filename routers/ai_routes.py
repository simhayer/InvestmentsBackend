from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from services.auth import get_current_user
from pydantic import BaseModel
from services.ai_service import analyze_investment_holding

router = APIRouter()

class HoldingInput(BaseModel):
    symbol: str
    name: str
    quantity: float
    purchase_price: float
    current_price: float
    type: str
    institution: str
    currency: str


@router.post("/analyze-holding")
async def analyze_holding(
    holding: HoldingInput,
    user=Depends(get_current_user),
):
    analysis = await run_in_threadpool(analyze_investment_holding, holding.model_dump())
    return {"insight": analysis}
