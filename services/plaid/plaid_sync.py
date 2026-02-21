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

# Canadian institutions / brokers where holdings are typically in CAD when Plaid omits currency
CAD_INSTITUTION_KEYWORDS = ("wealthsimple", "questrade", "wealth bar", "canada", "rbc direct", "td direct", "bmo investorline", "scotia itrade", "national bank", "desjardins")


def _get_iso_currency(obj: Optional[dict], *extra_keys: str) -> Optional[str]:
    """Extract ISO 4217 currency from a dict (holding, security, or balance). Tries common key names."""
    if not obj or not isinstance(obj, dict):
        return None
    keys = ["iso_currency_code", "isoCurrencyCode", "currency"] + list(extra_keys)
    for k in keys:
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().upper()
    return None


def _resolve_holding_currency(
    holding: dict,
    security: Optional[dict],
    institution_name: str,
    account: Optional[dict] = None,
    symbol: str = "",
    sec_type: str = "",
) -> str:
    """
    Resolve currency from the broker (Plaid). Prefer the security's price currency
    (trading currency) so it matches the purchase price (e.g. USD for VUG). Fall back
    to holding, then account balance, then institution inference.
    For crypto: prefer security/holding currency; default USD (crypto is usually quoted in USD).
    """
    holding_ccy = _get_iso_currency(holding)
    security_ccy = _get_iso_currency(security) if security else None
    account_balance = account.get("balance") if account and isinstance(account.get("balance"), dict) else None
    account_ccy = _get_iso_currency(account_balance) if account_balance else None
    # Plaid sometimes uses unofficial_currency_code for crypto
    unofficial = None
    if holding and isinstance(holding, dict):
        unofficial = (holding.get("unofficial_currency_code") or holding.get("unofficialCurrencyCode") or "").strip().upper()
    if security and isinstance(security, dict):
        unofficial = unofficial or (security.get("unofficial_currency_code") or security.get("unofficialCurrencyCode") or "").strip().upper()
    if unofficial and len(unofficial) >= 2:
        # Map common crypto symbols to USD (crypto is typically quoted in USD)
        if unofficial in ("BTC", "ETH", "USDT", "USDC", "BTCUSD", "ETHUSD"):
            unofficial = "USD"

    # Prefer security (trading/price currency) so stored currency matches purchase price
    raw = security_ccy or holding_ccy or (unofficial if unofficial else None) or account_ccy
    if raw and len(raw) >= 3:
        resolved = raw[:3].upper()
        if symbol and (holding_ccy != security_ccy or not holding_ccy):
            logger.info(
                "Plaid currency for %s (%s): holding=%s security=%s account=%s -> %s",
                symbol, sec_type or "n/a", holding_ccy, security_ccy, account_ccy, resolved,
            )
        return resolved

    # Crypto: default USD when broker sends no currency (crypto prices are typically USD)
    if (sec_type or "").lower() == "cryptocurrency":
        if symbol:
            logger.info("Plaid currency for crypto %s: none from API, defaulting USD", symbol)
        return "USD"

    # Infer CAD for Canadian brokers only when Plaid omits currency (equity/ETF)
    inst = (institution_name or "").lower()
    if any(kw in inst for kw in CAD_INSTITUTION_KEYWORDS):
        if symbol:
            logger.info("Plaid currency for %s: none from API, inferred CAD from institution %s", symbol, institution_name)
        return "CAD"
    if symbol:
        logger.info("Plaid currency for %s: none from API, defaulting USD", symbol)
    return "USD"


def _get(d: dict, *keys: str):
    """Get first present key from dict (supports both snake_case and camelCase from Plaid)."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _normalize_holdings(holdings: list, securities: list, accounts: list, institution_name: str) -> List[Dict]:
    # Support both snake_case and camelCase; Plaid SDK may vary
    security_lookup = {}
    for s in securities:
        sid = _get(s, "security_id", "securityId")
        if sid:
            security_lookup[sid] = s
    account_lookup = {}
    for a in accounts:
        aid = _get(a, "account_id", "accountId")
        if aid:
            account_lookup[aid] = a

    normalized = []
    for h in holdings:
        sec_id = _get(h, "security_id", "securityId")
        acc_id = _get(h, "account_id", "accountId")
        if not sec_id:
            continue
        sec = security_lookup.get(sec_id)
        sec_type_val = _get(sec, "type")
        if not sec or sec_type_val not in ALLOWED_TYPES:
            continue
        acc = account_lookup.get(acc_id) if acc_id else None

        qty = _get(h, "quantity")
        total_cost = _get(h, "cost_basis", "costBasis")
        unit_cost = safe_div(total_cost, qty)
        # Plaid often omits institution_price for Canadian/Wealthsimple; use cost per unit so value isn't $0
        inst_price = _get(h, "institution_price", "institutionPrice")
        current_price = inst_price if inst_price is not None else unit_cost

        current_value = (
            current_price * qty
            if (current_price is not None and qty is not None)
            else (total_cost if total_cost is not None else None)
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

        symbol_val = _get(sec, "ticker_symbol", "tickerSymbol") or ""
        currency = _resolve_holding_currency(h, sec, institution_name, acc, symbol=symbol_val, sec_type=sec_type_val or "")

        normalized.append({
            "externalId": sec_id,
            "accountId": acc_id,
            "accountName": (_get(acc, "name") if acc else "Unknown"),
            "institution": institution_name,

            "symbol": symbol_val,
            "name": _get(sec, "name") or "",
            "type": sec_type_val or "",
            "currency": currency,

            "quantity": qty,
            "currentPrice": current_price,
            "currentValue": current_value,
            "value": current_value,

            "purchaseAmountTotal": total_cost,
            "purchaseUnitPrice": unit_cost,
            "purchasePrice": unit_cost,

            "unrealizedPl": unrealized_pl,
            "unrealizedPlPct": unrealized_pl_pct,

            "previousClose": _get(h, "institution_price_datetime", "institutionPriceDatetime") or None,
            "priceStatus": _get(h, "price_source", "priceSource") or None,
        })
    return normalized


def _upsert_holdings(
    user_id: str,
    institution_name: str,
    plaid_holdings: List[Dict],
    db: Session,
    fallback_institution_name: Optional[str] = None,
):
    """Upsert holdings scoped to a specific institution (not all plaid holdings globally)."""
    existing = (
        db.query(Holding)
        .filter_by(user_id=str(user_id), source="plaid", institution=institution_name)
        .all()
    )
    if not existing and fallback_institution_name and fallback_institution_name.strip() != institution_name:
        existing = (
            db.query(Holding)
            .filter_by(user_id=str(user_id), source="plaid", institution=fallback_institution_name.strip())
            .all()
        )
        if existing:
            logger.info("Plaid upsert: matched existing rows by fallback institution name %s", fallback_institution_name)
    # Key by (account_name, external_id) so same security in TFSA vs Non-registered = separate rows
    existing_map = {(h.account_name or "", h.external_id): h for h in existing}
    seen_keys = set()
    updates = 0
    inserts = 0

    for ph in plaid_holdings:
        sec_id = ph.get("externalId") or ph.get("external_id")
        account_name = (ph.get("accountName") or ph.get("account_name") or "").strip()
        if not sec_id:
            logger.warning("Skipping holding without externalId: %s", ph)
            continue

        row_key = (account_name, sec_id)
        seen_keys.add(row_key)

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
        institution = ph.get("institution") or ""
        # Persist correct ISO 4217 currency (e.g. CAD, USD) to Supabase; accept both keys from normalized payload
        currency = _get_iso_currency(ph) or (ph.get("currency") or "").strip().upper()
        if not currency or len(currency) < 3:
            if (sec_type or "").lower() == "cryptocurrency":
                currency = "USD"
            else:
                currency = "CAD" if any(kw in (institution or "").lower() for kw in CAD_INSTITUTION_KEYWORDS) else "USD"
        currency = currency[:3].upper()  # ensure exactly 3-letter ISO code

        if row_key in existing_map:
            h = existing_map[row_key]
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
            updates += 1
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
            inserts += 1

    if plaid_holdings:
        logger.info(
            "Plaid upsert: institution=%s updated=%s inserted=%s currencies=%s",
            institution_name,
            updates,
            inserts,
            [ph.get("currency") for ph in plaid_holdings[:5]],
        )
    for h in existing:
        if (h.account_name or "", h.external_id) not in seen_keys:
            db.delete(h)


def _fetch_holdings_with_retry(access_token: str, max_retries: int = 5) -> dict:
    """
    Call investments/holdings/get with exponential back-off for PRODUCT_NOT_READY.
    Some institutions (e.g. Canadian brokerages) can take longer to prepare data.
    """
    for attempt in range(max_retries):
        try:
            req = InvestmentsHoldingsGetRequest(access_token=access_token)
            return client.investments_holdings_get(req).to_dict()
        except ApiException as e:
            is_not_ready = e.status == 400 and "PRODUCT_NOT_READY" in str(e.body)
            if is_not_ready and attempt < max_retries - 1:
                wait = (2 ** attempt) + 2  # 3s, 4s, 6s, 8s so first wait is a bit longer
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
    # Use stored institution name first so we match existing DB rows (currency updates apply)
    institution_name = (token_entry.institution_name or response.get("item", {}).get("institution_name") or "Unknown").strip()
    if not token_entry.institution_name:
        token_entry.institution_name = institution_name  # persist so next sync finds same rows

    normalized = _normalize_holdings(holdings, securities, accounts, institution_name)
    if normalized:
        logger.info(
            "Plaid sync: institution=%s holdings=%s sample_currency=%s",
            institution_name,
            len(normalized),
            normalized[0].get("currency") if normalized else None,
        )
    api_institution = response.get("item", {}).get("institution_name") or ""
    _upsert_holdings(
        str(token_entry.user_id),
        institution_name,
        normalized,
        db,
        fallback_institution_name=api_institution if api_institution != institution_name else None,
    )

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
