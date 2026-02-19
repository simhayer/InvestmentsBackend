import logging
import os
import traceback

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.access_token import UserAccess
from models.user import User
from services.plaid.plaid_config import client
from services.plaid.plaid_service import get_connections, remove_connection
from services.plaid.plaid_sync import sync_all_connections, sync_by_item_id
from services.supabase_auth import get_current_db_user

logger = logging.getLogger(__name__)
router = APIRouter()

PLAID_WEBHOOK_URL = os.getenv("PLAID_WEBHOOK_URL")

# ----------- LINK TOKEN ----------------

@router.post("/create-link-token")
async def create_link_token(user: User = Depends(get_current_db_user)):

    try:
        kwargs = dict(
            products=[Products("investments")],
            client_name="Investment Tracker",
            country_codes=[CountryCode("US"), CountryCode("CA")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=str(user.id)),
        )
        if PLAID_WEBHOOK_URL:
            kwargs["webhook"] = PLAID_WEBHOOK_URL

        request = LinkTokenCreateRequest(**kwargs)
        response = client.link_token_create(request)
        return {"link_token": response.link_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating link token: {e}")


# ----------- EXCHANGE TOKEN ----------------

@router.post("/exchange-token")
async def exchange_token(
    public_token: str = Body(...),
    institution_id: str = Body(None),
    institution_name: str = Body(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
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
    user: User = Depends(get_current_db_user),
):
    token_entries = db.query(UserAccess).filter_by(user_id=str(user.id)).all()
    if not token_entries:
        raise HTTPException(status_code=404, detail="User access token not found")

    all_normalized = sync_all_connections(user, db)

    return {
        "message": "Holdings synced",
        "count": len(all_normalized),
        "holdings": all_normalized,
    }


# ----------- PLAID WEBHOOK ----------------
@router.post("/webhook")
async def plaid_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    webhook_type = body.get("webhook_type", "")
    webhook_code = body.get("webhook_code", "")
    item_id = body.get("item_id", "")

    logger.info("Plaid webhook received: type=%s code=%s item=%s", webhook_type, webhook_code, item_id)

    if not item_id:
        return {"received": True}

    if webhook_type == "HOLDINGS" and webhook_code in ("DEFAULT_UPDATE", "NEW_HOLDINGS_AVAILABLE"):
        sync_by_item_id(item_id, db)

    elif webhook_type == "INVESTMENTS_TRANSACTIONS" and webhook_code == "DEFAULT_UPDATE":
        sync_by_item_id(item_id, db)

    elif webhook_type == "ITEM":
        token_entry = db.query(UserAccess).filter_by(item_id=item_id).first()
        if token_entry:
            error = body.get("error") or {}
            error_code = error.get("error_code", "")
            if webhook_code == "ERROR" and error_code == "ITEM_LOGIN_REQUIRED":
                token_entry.status = "error"
                db.commit()
                logger.warning("Connection %s requires re-authentication", token_entry.id)
            elif webhook_code == "PENDING_EXPIRATION":
                token_entry.status = "error"
                db.commit()
                logger.warning("Connection %s access consent expiring soon", token_entry.id)

    return {"received": True}

class InstitutionOut(BaseModel):
    id: str
    institution_name: str
    institution_id: str
    status: str = "connected"
    created_at: str
    synced_at: str | None = None

@router.get("/institutions", response_model=List[InstitutionOut])
async def get_connected_institutions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        return get_connections(str(user.id), db)
    except Exception as e:
        logger.exception("Error fetching institutions")
        raise HTTPException(status_code=500, detail="Failed to fetch institutions")


@router.delete("/institutions/{connection_id}")
async def delete_connection(
    connection_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        remove_connection(connection_id, str(user.id), db)
        return {"detail": "Connection removed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error removing connection %s", connection_id)
        raise HTTPException(status_code=500, detail="Failed to remove connection")