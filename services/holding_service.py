from sqlalchemy.orm import Session
from models.holding import Holding
from typing import Any

def get_all_holdings(user_id: str, db: Session) -> list[dict[str, Any]]:
    """
    Get all holdings for a user.
    """
    holdings = db.query(Holding).filter_by(user_id=user_id).all()

    return [
        {
            "id": h.id,
            "symbol": h.symbol,
            "name": h.name,
            "type": h.type,
            "quantity": h.quantity,
            "purchase_price": h.purchase_price,
            "current_price": h.current_price,
            "value": h.value,
            "currency": h.currency,
            "institution": h.institution,
            "account_name": h.account_name,
            "source": h.source,
        }
        for h in holdings
    ]