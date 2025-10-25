from sqlalchemy.orm import Session
from sqlalchemy import delete
from models.market_overview import MarketOverviewLatest, MarketOverviewHistory
from typing import Any, Dict, Optional
from datetime import datetime

Json = Dict[str, Any]

def db_read_latest(db: Session) -> Optional[Json]:
    record = db.query(MarketOverviewLatest).order_by(MarketOverviewLatest.id.desc()).first()
    if not record:
        return None
    return {
        "symbols": record.symbols,
        "items": record.items,
        "fetched_at": record.fetched_at.isoformat() if record.fetched_at else None,
    }

def db_upsert_latest(db: Session, payload: Json) -> None:
    db.execute(delete(MarketOverviewLatest))  # enforce one row
    latest = MarketOverviewLatest(
        symbols=payload["symbols"],
        items=payload["items"],
        fetched_at=datetime.utcnow(),
    )
    db.add(latest)
    db.commit()

def db_append_history(db: Session, payload: Json) -> None:
    hist = MarketOverviewHistory(
        symbols=payload["symbols"],
        items=payload["items"],
    )
    db.add(hist)
    db.commit()
