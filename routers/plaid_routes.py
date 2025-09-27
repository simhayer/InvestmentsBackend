import os
from fastapi import APIRouter, HTTPException, Body
from plaid import Configuration, ApiClient, Environment
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from database import get_db
from models.access_token import UserAccess
from sqlalchemy.orm import Session
from fastapi import Depends
from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest
from pydantic import BaseModel
from models.holding import Holding
from typing import List, Dict
from services.auth_service import get_current_user

# Plaid setup
configuration = Configuration(
    host=os.getenv("PLAID_ENV", "sandbox").lower() == "production" and Environment.Production or Environment.Sandbox,
    api_key={
        "clientId": os.getenv("PLAID_CLIENT_ID"),
        "secret": os.getenv("PLAID_SECRET"),
    },
)
client = plaid_api.PlaidApi(ApiClient(configuration))

router = APIRouter()

# ----------- LINK TOKEN ----------------

@router.post("/create-link-token")
async def create_link_token(user=Depends(get_current_user)):
    try:
        request = LinkTokenCreateRequest(
            products=[Products("investments")],
            client_name="Investment Tracker",
            country_codes=[CountryCode("US"), CountryCode("CA")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=str(user.id)),
        )
        response = client.link_token_create(request)
        return {"link_token": response.link_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating link token: {e}")


# ----------- EXCHANGE TOKEN ----------------
import logging
logger = logging.getLogger(__name__)
import traceback

@router.post("/exchange-token")
async def exchange_token(
    public_token: str = Body(...),
    institution_id: str = Body(None),
    institution_name: str = Body(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = client.item_public_token_exchange(request)

        access_token = response.access_token
        item_id = response.item_id

        token_entry = db.query(UserAccess).filter_by(
            user_id=str(user.id), institution_id=institution_id
        ).first()

        if token_entry:
            token_entry.access_token = access_token
            token_entry.item_id = item_id
            token_entry.institution_name = institution_name
        else:
            db.add(UserAccess(
                user_id=str(user.id),
                access_token=access_token,
                item_id=item_id,
                institution_id=institution_id,
                institution_name=institution_name,
        ))

        db.commit()

        return {"access_token": access_token}

    except Exception as e:
        logger.error("Token exchange failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")


# ----------- SYNC HOLDINGS ----------------

@router.get("/investments")
async def get_investments(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        token_entry = db.query(UserAccess).filter_by(user_id=str(user.id)).first()
        if not token_entry:
            raise HTTPException(status_code=404, detail="User access token not found")

        request = InvestmentsHoldingsGetRequest(access_token=token_entry.access_token)
        response = client.investments_holdings_get(request).to_dict()

        holdings = response.get("holdings", [])
        securities = response.get("securities", [])
        accounts = response.get("accounts", [])
        institution_name = response.get("item", {}).get("institution_name", "Unknown")

        # Create lookup maps
        security_lookup = {sec["security_id"]: sec for sec in securities}
        account_lookup = {acc["account_id"]: acc for acc in accounts}
        investment_types = {"equity", "etf", "cryptocurrency"}

        # Format holdings
        plaid_holdings = []
        for h in holdings:
            sec = security_lookup.get(h["security_id"])
            acc = account_lookup.get(h["account_id"])
            purchase_price = h.get("cost_basis")
            if not sec or sec.get("type") not in investment_types:
                continue

            plaid_holdings.append({
                "external_id": h["security_id"],
                "symbol": sec.get("ticker_symbol"),
                "name": sec.get("name"),
                "type": sec.get("type"),
                "quantity": h.get("quantity"),
                "purchase_price": purchase_price,
                "current_price": h.get("institution_price"),
                "value": h.get("institution_value"),
                "currency": h.get("iso_currency_code"),
                "account_name": acc.get("name") if acc else "Unknown",
                "institution": institution_name,
            })

        sync_plaid_holdings(user.id, plaid_holdings, db)
        return {"message": "Holdings synced", "count": len(plaid_holdings), "all_holdings": response}

    except Exception as e:
        #traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch investments: {e}")

def sync_plaid_holdings(user_id: int, plaid_holdings: List[Dict], db: Session):
    # Step 1: Fetch existing holdings
    existing_holdings = db.query(Holding).filter_by(user_id=str(user_id), source="plaid").all()

    existing_map = {h.external_id: h for h in existing_holdings}

    seen_ids = set()

    # Step 2: Update or insert
    for ph in plaid_holdings:
        sec_id = ph["external_id"]
        seen_ids.add(sec_id)

        if sec_id in existing_map:
            h = existing_map[sec_id]
            h.symbol = ph["symbol"]
            h.name = ph["name"]
            h.quantity = ph["quantity"]
            h.current_price = ph["current_price"]
            h.purchase_price = ph["purchase_price"]
            h.value = ph["value"]
            h.account_name = ph["account_name"]
            h.institution = ph["institution"]
            h.currency = ph["currency"]
        else:
            new_holding = Holding(
                user_id=str(user_id),
                source="plaid",
                external_id=sec_id,
                symbol=ph["symbol"],
                name=ph["name"],
                type=ph["type"],
                quantity=ph["quantity"],
                current_price=ph["current_price"],
                purchase_price=ph["purchase_price"],
                value=ph["value"],
                account_name=ph["account_name"],
                institution=ph["institution"],
                currency=ph["currency"],
            )
            db.add(new_holding)

    # Step 3 (optional): Delete stale Plaid holdings
    for h in existing_holdings:
        if h.external_id not in seen_ids:
            db.delete(h)

    db.commit()

class InstitutionOut(BaseModel):
    institution_name: str
    institution_id: str
    created_at: str

@router.get("/institutions", response_model=List[InstitutionOut])
async def get_connected_institutions(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        institutions = db.query(UserAccess).filter(UserAccess.user_id == str(user.id)).all()

        return [
            {
                "institution_name": ua.institution_name,
                "institution_id": ua.institution_id,
                "created_at": ua.created_at.isoformat(),
            }
            for ua in institutions
            if ua.institution_id and ua.institution_name
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch institutions")