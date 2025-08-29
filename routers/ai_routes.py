from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from services.auth_service import get_current_user
from services.ai_service import analyze_investment_holding, analyze_investment_portfolio, analyze_investment_holding_forecast, analyze_investment_holding_perplexity
from schemas.holding import HoldingInputPydantic
from utils.converters import to_holding_dict
from services.holding_service import get_all_holdings
from sqlalchemy.orm import Session
from database import get_db

router = APIRouter()

# @router.post("/analyze-holding")
# async def analyze_holding(
#     holding: HoldingInputPydantic,
#     user=Depends(get_current_user),
# ):
#     # analyze_investment_holding is a *sync* function
#     analysis = await run_in_threadpool(analyze_investment_holding, to_holding_dict(holding))
#     return {"insight": analysis}


#trying forecase 
@router.post("/analyze-holding")
async def analyze_holding(
    holding: HoldingInputPydantic,
    user=Depends(get_current_user),
):
    # analyze_investment_holding is a *sync* function
    analysis = await run_in_threadpool(analyze_investment_holding_perplexity, to_holding_dict(holding))
    return {"insight": analysis}

@router.post("/analyze-portfolio")
async def analyze_portfolio_endpoint(
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    holdings = get_all_holdings(user.id, db)
    return await run_in_threadpool(analyze_investment_portfolio, holdings)