from fastapi import APIRouter

from services.market_service import get_market_overview_items

router = APIRouter()

@router.get("/overview")
async def get_market_overview():
    top_items = await get_market_overview_items()
    return {
        "message": "Market overview data",
        "data": {
            "top_items": top_items
        }
    }