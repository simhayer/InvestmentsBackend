from fastapi import BackgroundTasks, APIRouter, Depends, Request
from services.filings.filing_service import FilingService
from services.supabase_auth import get_current_db_user
from middleware.rate_limit import limiter

router = APIRouter()

@router.post("/store/{symbol}")
@limiter.limit("5/minute")
async def trigger_filings_store(
    request: Request,
    symbol: str,
    background_tasks: BackgroundTasks,
    _user=Depends(get_current_db_user),
):
    background_tasks.add_task(FilingService().process_company_filings_task, symbol)
    return {
        "status": "Accepted",
        "message": f"Processing filings for {symbol} in the background."
    }