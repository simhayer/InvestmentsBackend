from fastapi import BackgroundTasks, Depends, APIRouter
from sqlalchemy.orm import Session
from database import get_db
from services.filings.filing_service import FilingService

router = APIRouter()

@router.post("/store/{symbol}")
async def trigger_analysis(
    symbol: str, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    # Add the task to the background queue
    background_tasks.add_task(FilingService().process_company_filings_task, symbol, db)
    
    return {
        "status": "Accepted", 
        "message": f"Processing filings for {symbol} in the background. Insights will appear shortly."
    }