# services/plaid_service.py

from sqlalchemy.orm import Session
from models.access_token import UserAccess
from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest
from plaid_config import client  # <- Make sure you import your configured client
from typing import List, Dict


def fetch_plaid_holdings_for_user(user_id: str, db: Session) -> List[Dict]:
    # access_tokens = db.query(UserAccess).filter_by(user_id=user_id).all()
    access_tokens = db.query(UserAccess).filter_by(user_id=str(user_id)).all()
    all_holdings = []

    for token_entry in access_tokens:
        try:
            request = InvestmentsHoldingsGetRequest(access_token=token_entry.access_token)
            response = client.investments_holdings_get(request)
            plaid_data = response.to_dict()

            for h in plaid_data.get("holdings", []):
                all_holdings.append({
                    "source": "plaid",
                    "institution": token_entry.institution_name,
                    "symbol": h.get("security_id"),
                    "quantity": h.get("quantity"),
                    "current_price": h.get("institution_price"),
                    "value": h.get("institution_value"),
                })

        except Exception as e:
            print(f"❌ Error fetching from {token_entry.institution_name}: {e}")
            continue

    return all_holdings
