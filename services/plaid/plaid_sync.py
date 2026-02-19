"""
Per-item sync logic for Plaid investment holdings.

Used by both the manual /investments endpoint and the Plaid webhook handler.
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

from plaid.exceptions import ApiException
from sqlalchemy.orm import Session
from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest

from models.access_token import UserAccess
from models.holding import Holding
from services.plaid.plaid_config import client
from services.currency_service import maybe_auto_set_user_base_currency
from models.user import User
from utils.common_helpers import safe_div, num

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"equity", "etf", "cryptocurrency"}


def _normalize_holdings(holdings: list, securities: list, accounts: list, institution_name: str) -> List[Dict]:
    security_lookup = {s["security_id"]: s for s in securities}
    account_lookup = {a["account_id"]: a for a in accounts}

    normalized = []
    for h in holdings:
        sec = security_lookup.get(h["security_id"])
        if not sec or sec.get("type") not in ALLOWED_TYPES:
            continue
        acc = account_lookup.get(h["account_id"])

        qty = h.get("quantity")
        current_price = h.get("institution_price")
        total_cost = h.get("cost_basis")
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
            "externalId": h.get("security_id"),
            "accountId": h.get("account_id"),
            "accountName": (acc.get("name") if acc else "Unknown"),
            "institution": institution_name,

            "symbol": sec.get("ticker_symbol"),
            "name": sec.get("name"),
            "type": sec.get("type"),
            "currency": h.get("iso_currency_code"),

            "quantity": qty,
            "currentPrice": current_price,
            "currentValue": current_value,
            "value": current_value,

            "purchaseAmountTotal": total_cost,
            "purchaseUnitPrice": unit_cost,
            "purchasePrice": unit_cost,

            "unrealizedPl": unrealized_pl,
            "unrealizedPlPct": unrealized_pl_pct,

            "previousClose": h.get("institution_price_datetime") or None,
            "priceStatus": h.get("price_source") or None,
        })
    return normalized


def _upsert_holdings(user_id: str, institution_name: str, plaid_holdings: List[Dict], db: Session):
    """Upsert holdings scoped to a specific institution (not all plaid holdings globally)."""
    existing = (
        db.query(Holding)
        .filter_by(user_id=str(user_id), source="plaid", institution=institution_name)
        .all()
    )
    existing_map = {h.external_id: h for h in existing}
    seen_ids = set()

    for ph in plaid_holdings:
        sec_id = ph.get("externalId") or ph.get("external_id")
        if not sec_id:
            logger.warning("Skipping holding without externalId: %s", ph)
            continue

        seen_ids.add(sec_id)

        qty = num(ph.get("quantity"))
        current_price = num(ph.get("currentPrice") or ph.get("current_price"))
        total_cost = num(
            ph.get("purchaseAmountTotal")
            or ph.get("purchase_amount_total")
            or ph.get("purchasePrice")
            or ph.get("purchase_price")
        )
        current_value = num(ph.get("currentValue") or ph.get("value"))
        if current_value is None and current_price is not None and qty is not None:
            current_value = current_price * qty

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
        account_name = ph.get("accountName") or ph.get("account_name") or ""
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
            h.purchase_amount_total = total_cost
            h.purchase_unit_price = unit_cost
            h.unrealized_pl = unrealized_pl
            h.unrealized_pl_pct = unrealized_pl_pct
            h.purchase_price = unit_cost or 0.0
            h.value = current_value or 0.0
            h.account_name = account_name
            h.institution = institution
            h.currency = currency
        else:
            db.add(Holding(
                user_id=str(user_id),
                source="plaid",
                external_id=sec_id,
                symbol=symbol,
                name=name,
                type=sec_type,
                quantity=qty or 0.0,
                current_price=current_price or 0.0,
                current_value=current_value,
                purchase_amount_total=total_cost,
                purchase_unit_price=unit_cost,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,
                purchase_price=unit_cost or 0.0,
                value=current_value or 0.0,
                account_name=account_name,
                institution=institution,
                currency=currency,
            ))

    for h in existing:
        if h.external_id not in seen_ids:
            db.delete(h)


def _fetch_holdings_with_retry(access_token: str, max_retries: int = 3) -> dict:
    """Call investments/holdings/get with exponential back-off for PRODUCT_NOT_READY."""
    for attempt in range(max_retries):
        try:
            req = InvestmentsHoldingsGetRequest(access_token=access_token)
            return client.investments_holdings_get(req).to_dict()
        except ApiException as e:
            is_not_ready = e.status == 400 and "PRODUCT_NOT_READY" in str(e.body)
            if is_not_ready and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info(
                    "PRODUCT_NOT_READY — retrying in %ds (attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                continue
            raise


def sync_single_connection(token_entry: UserAccess, db: Session) -> List[Dict]:
    """
    Fetch holdings from Plaid for a single connection and persist them.
    Returns the normalized holdings list.
    """
    response = _fetch_holdings_with_retry(token_entry.access_token)

    holdings = response.get("holdings", [])
    securities = response.get("securities", [])
    accounts = response.get("accounts", [])
    institution_name = response.get("item", {}).get("institution_name") or token_entry.institution_name or "Unknown"

    normalized = _normalize_holdings(holdings, securities, accounts, institution_name)
    _upsert_holdings(str(token_entry.user_id), institution_name, normalized, db)

    token_entry.synced_at = datetime.now(timezone.utc)
    token_entry.status = "connected"
    db.commit()

    return normalized


def sync_all_connections(user: User, db: Session) -> List[Dict]:
    """
    Sync holdings for ALL of a user's Plaid connections.
    Returns the combined normalized holdings list.
    """
    token_entries = db.query(UserAccess).filter_by(user_id=str(user.id)).all()
    if not token_entries:
        return []

    all_normalized: List[Dict] = []
    for entry in token_entries:
        try:
            normalized = sync_single_connection(entry, db)
            all_normalized.extend(normalized)
        except Exception:
            logger.exception("Failed to sync connection %s (%s)", entry.id, entry.institution_name)
            entry.status = "error"
            db.commit()

    if all_normalized:
        maybe_auto_set_user_base_currency(user, all_normalized, db)

    return all_normalized


def sync_by_item_id(item_id: str, db: Session) -> Optional[List[Dict]]:
    """
    Sync holdings for a specific Plaid item_id (used by webhook handler).
    Returns normalized holdings, or None if item not found.
    """
    token_entry = db.query(UserAccess).filter_by(item_id=item_id).first()
    if not token_entry:
        logger.warning("Webhook sync: no UserAccess found for item_id=%s", item_id)
        return None

    try:
        return sync_single_connection(token_entry, db)
    except Exception:
        logger.exception("Webhook sync failed for item_id=%s", item_id)
        token_entry.status = "error"
        db.commit()
        return None
