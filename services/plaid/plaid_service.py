# services/plaid_service.py

from sqlalchemy.orm import Session
from models.access_token import UserAccess
from typing import List, Dict

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
