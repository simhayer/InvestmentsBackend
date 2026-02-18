# services/plaid_service.py

import logging
from sqlalchemy.orm import Session
from models.access_token import UserAccess
from models.holding import Holding
from typing import List, Dict
from plaid.model.item_remove_request import ItemRemoveRequest
from services.plaid.plaid_config import client

logger = logging.getLogger(__name__)

def get_connections(userId: str, db: Session) -> List[Dict]:
    institutions = db.query(UserAccess).filter(UserAccess.user_id == str(userId)).all()

    return [
        {
            "id": ua.id,
            "institution_name": ua.institution_name,
            "institution_id": ua.institution_id,
            "created_at": ua.created_at.isoformat(),
            "synced_at": ua.synced_at.isoformat() if ua.synced_at else None,
        }
        for ua in institutions
        if ua.institution_id and ua.institution_name
    ]


def remove_connection(connection_id: str, user_id: str, db: Session) -> None:
    token_entry = (
        db.query(UserAccess)
        .filter_by(id=connection_id, user_id=str(user_id))
        .first()
    )
    if not token_entry:
        raise ValueError("Connection not found")

    # Revoke the access token on Plaid's side (best-effort)
    try:
        client.item_remove(ItemRemoveRequest(access_token=token_entry.access_token))
    except Exception:
        logger.warning("Plaid item_remove failed for connection %s; continuing local cleanup", connection_id)

    # Delete associated Plaid-sourced holdings for this institution
    db.query(Holding).filter_by(
        user_id=str(user_id),
        source="plaid",
        institution=token_entry.institution_name,
    ).delete(synchronize_session="fetch")

    db.delete(token_entry)
    db.commit()
