from fastapi import BackgroundTasks, APIRouter
from services.filings.filing_service import FilingService

router = APIRouter()

@router.post("/store/{symbol}")
async def trigger_filings_store(symbol: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(FilingService().process_company_filings_task, symbol)
    return {
        "status": "Accepted",
        "message": f"Processing filings for {symbol} in the background."
    }