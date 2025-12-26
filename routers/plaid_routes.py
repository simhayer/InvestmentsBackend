from datetime import datetime, timezone
import os
from fastapi import APIRouter, HTTPException, Body
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
from services.plaid_service import get_connections
from utils.common_helpers import safe_div, num
from models.user import User
from services.supabase_auth import get_current_db_user
from services.currency_service import maybe_auto_set_user_base_currency
from plaid_config import client
import logging
logger = logging.getLogger(__name__)
import traceback
router = APIRouter()

# ----------- LINK TOKEN ----------------

@router.post("/create-link-token")
async def create_link_token(user: User = Depends(get_current_db_user)):

    try:
        request = LinkTokenCreateRequest(
            products=[Products("investments")],
            client_name="Investment Tracker",
            country_codes=[CountryCode("US"), CountryCode("CA")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=str(user.id))
        )
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
    token_entry = db.query(UserAccess).filter_by(user_id=str(user.id)).first()
    if not token_entry:
        raise HTTPException(status_code=404, detail="User access token not found")

    request = InvestmentsHoldingsGetRequest(access_token=token_entry.access_token)
    response = client.investments_holdings_get(request).to_dict()

    holdings = response.get("holdings", [])
    securities = response.get("securities", [])
    accounts = response.get("accounts", [])
    institution_name = response.get("item", {}).get("institution_name", "Unknown")

    security_lookup = {s["security_id"]: s for s in securities}
    account_lookup = {a["account_id"]: a for a in accounts}

    allowed_types = {"equity", "etf", "cryptocurrency"}

    normalized = []
    for h in holdings:
        sec = security_lookup.get(h["security_id"])
        if not sec or sec.get("type") not in allowed_types:
            continue
        acc = account_lookup.get(h["account_id"])

        qty = h.get("quantity")
        current_price = h.get("institution_price")
        total_cost = h.get("cost_basis")  # TOTAL dollars spent (ACB)
        unit_cost = safe_div(total_cost, qty)

        current_value = (
            current_price * qty
            if (current_price is not None and qty is not None)
            else None
        )
        unrealized_pl = (
            current_value - total_cost
            if (current_value is not None and total_cost is not None)
            else None
        )
        unrealized_pl_pct = safe_div(current_value, total_cost)
        unrealized_pl_pct = (
            (unrealized_pl_pct - 1) * 100
            if unrealized_pl_pct is not None
            else None
        )

        normalized.append({
            # identifiers
            "externalId": h.get("security_id"),
            "accountId": h.get("account_id"),
            "accountName": (acc.get("name") if acc else "Unknown"),
            "institution": institution_name,

            # security
            "symbol": sec.get("ticker_symbol"),
            "name": sec.get("name"),
            "type": sec.get("type"),
            "currency": h.get("iso_currency_code"),

            # positions & pricing
            "quantity": qty,
            "currentPrice": current_price,       # per-unit (as provided by Plaid)
            "currentValue": current_value,       # computed total
            "value": current_value,              # alias used by DB sync

            # cost basis
            "purchaseAmountTotal": total_cost,   # TOTAL (Plaid cost_basis)
            "purchaseUnitPrice": unit_cost,      # derived per-unit
            "purchasePrice": unit_cost,          # legacy alias = per-unit (for old UI)

            # P/L (totals-based)
            "unrealizedPl": unrealized_pl,
            "unrealizedPlPct": unrealized_pl_pct,

            # optional passthroughs
            "previousClose": h.get("institution_price_datetime") or None,
            "priceStatus": h.get("price_source") or None,
        })

    # Persist to DB
    sync_plaid_holdings(str(user.id), normalized, db)

    # Auto-set base currency (only if not manual)
    maybe_auto_set_user_base_currency(user, normalized, db)

    return {
        "message": "Holdings synced",
        "count": len(normalized),
        "holdings": normalized,
        "raw": response,  # keep for now while debugging; remove later
    }
def sync_plaid_holdings(user_id: str, plaid_holdings: List[Dict], db: Session):
    # Step 1: Fetch existing holdings
    existing_holdings = (
        db.query(Holding)
        .filter_by(user_id=str(user_id), source="plaid")
        .all()
    )
    existing_map = {h.external_id: h for h in existing_holdings}
    seen_ids = set()

    # Step 2: Upsert
    for ph in plaid_holdings:
        # ID: support both camelCase and snake_case just in case
        sec_id = ph.get("externalId") or ph.get("external_id")
        if not sec_id:
            # Don't crash on bad data, just skip and log
            print("Skipping holding without externalId/external_id:", ph)
            continue

        seen_ids.add(sec_id)

        # Core numbers
        qty = num(ph.get("quantity"))
        current_price = num(ph.get("currentPrice") or ph.get("current_price"))

        # Total cost: prefer the explicit total, fall back to legacy names
        total_cost = num(
            ph.get("purchaseAmountTotal")
            or ph.get("purchase_amount_total")
            or ph.get("purchasePrice")
            or ph.get("purchase_price")
        )

        # Value (market value)
        current_value = num(
            ph.get("currentValue")
            or ph.get("value")
        )
        if current_value is None and current_price is not None and qty is not None:
            current_value = current_price * qty

        # Derivations
        unit_cost = safe_div(total_cost, qty)
        unrealized_pl = (
            current_value - total_cost
            if (current_value is not None and total_cost is not None)
            else None
        )
        unrealized_pl_pct = None
        if total_cost is not None and total_cost > 0 and current_value is not None:
            unrealized_pl_pct = ((current_value / total_cost) - 1.0) * 100.0

        symbol = ph.get("symbol") or ""
        name = ph.get("name") or ""
        sec_type = ph.get("type") or ""
        account_name = (
            ph.get("accountName")
            or ph.get("account_name")
            or ""
        )
        institution = ph.get("institution") or ""
        currency = ph.get("currency") or ""

        if sec_id in existing_map:
            h = existing_map[sec_id]
            h.symbol = symbol
            h.name = name
            h.type = sec_type
            h.quantity = qty or 0.0
            h.current_price = current_price or 0.0
            h.current_value = current_value

            # NEW explicit fields (if you added these columns)
            h.purchase_amount_total = total_cost
            h.purchase_unit_price = unit_cost
            h.unrealized_pl = unrealized_pl
            h.unrealized_pl_pct = unrealized_pl_pct

            # Legacy field: keep as **unit** price for old UI
            h.purchase_price = unit_cost or 0.0

            # Total value
            h.value = current_value or 0.0

            h.account_name = account_name
            h.institution = institution
            h.currency = currency
        else:
            new_holding = Holding(
                user_id=str(user_id),
                source="plaid",
                external_id=sec_id,
                symbol=symbol,
                name=name,
                type=sec_type,
                quantity=qty or 0.0,
                current_price=current_price or 0.0,
                current_value=current_value,

                # NEW explicit fields
                purchase_amount_total=total_cost,
                purchase_unit_price=unit_cost,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,

                # Legacy unit alias
                purchase_price=unit_cost or 0.0,

                value=current_value or 0.0,
                account_name=account_name,
                institution=institution,
                currency=currency,
            )
            db.add(new_holding)

    # Step 3: Delete stale Plaid holdings (optional)
    for h in existing_holdings:
        if h.external_id not in seen_ids:
            db.delete(h)

    # Step 4: mark tokens synced
    tokens = db.query(UserAccess).filter_by(user_id=str(user_id)).all()
    for t in tokens:
        t.synced_at = datetime.now(timezone.utc)

    db.commit()

class InstitutionOut(BaseModel):
    id: str
    institution_name: str
    institution_id: str
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
        print("Error fetching institutions:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch institutions")