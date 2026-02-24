from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from typing import Literal, Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from models.user_subscription import UserSubscription
from services.supabase_auth import get_current_db_user

router = APIRouter()
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
PRICE_PREMIUM = os.environ.get("STRIPE_PRICE_PREMIUM")
PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

Plan = Literal["premium", "pro"]

REFUND_WINDOW_DAYS = int(os.environ.get("REFUND_WINDOW_DAYS", "7"))


def get_or_create_sub(db: Session, user: User) -> UserSubscription:
    sub = db.query(UserSubscription).filter_by(user_id=user.id).first()
    if sub:
        return sub
    sub = UserSubscription(user_id=user.id, plan="free", status="free")
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def ensure_customer(db: Session, user: User) -> str:
    row = get_or_create_sub(db, user)
    if row.stripe_customer_id:
        return row.stripe_customer_id

    customer = stripe.Customer.create(
        email=user.email,
        metadata={"user_id": str(user.id), "supabase_user_id": user.supabase_user_id},
    )
    row.stripe_customer_id = customer["id"]
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.stripe_customer_id or ""


def price_for_plan(plan: Plan) -> str:
    if plan == "premium":
        if not PRICE_PREMIUM:
            raise HTTPException(500, "STRIPE_PRICE_PREMIUM missing")
        return PRICE_PREMIUM
    if plan == "pro":
        if not PRICE_PRO:
            raise HTTPException(500, "STRIPE_PRICE_PRO missing")
        return PRICE_PRO
    raise HTTPException(400, "Invalid plan")


class CheckoutIn(BaseModel):
    plan: Plan


class CheckoutOut(BaseModel):
    url: str


@router.post("/checkout-session", response_model=CheckoutOut)
def create_checkout_session(
    payload: CheckoutIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    customer_id = ensure_customer(db, user)
    price_id = price_for_plan(payload.plan)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/settings?checkout=success",
            cancel_url=f"{FRONTEND_URL}/settings?checkout=cancel",
            subscription_data={},
            allow_promotion_codes=True,
        )
        logger.info("checkout_session_created user_id=%s plan=%s customer_id=%s", user.id, payload.plan, customer_id)
        return {"url": session["url"]}
    except Exception as e:
        logger.exception("checkout_session_failed user_id=%s: %s", user.id, e)
        raise HTTPException(400, detail="Could not create checkout session")


class PortalOut(BaseModel):
    url: str


@router.post("/portal-session", response_model=PortalOut)
def create_portal_session(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    row = get_or_create_sub(db, user)
    if not row.stripe_customer_id:
        raise HTTPException(400, detail="No Stripe customer yet")

    session = stripe.billing_portal.Session.create(
        customer=row.stripe_customer_id,
        return_url=f"{FRONTEND_URL}/settings",
    )
    logger.info("portal_session_created user_id=%s", user.id)
    return {"url": session["url"]}


class SubOut(BaseModel):
    plan: str
    status: str
    current_period_end: Optional[datetime] = None
    trial_end: Optional[datetime] = None


@router.get("/me", response_model=SubOut)
def get_my_subscription(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    row = get_or_create_sub(db, user)
    return SubOut(
        plan=row.plan,
        status=row.status,
        current_period_end=row.current_period_end,
        trial_end=row.trial_end,
    )


class UsageLimitItem(BaseModel):
    feature: str
    used: int
    limit: int  # -1 = unlimited


class UsageOut(BaseModel):
    plan: str
    usage: list[UsageLimitItem]


FEATURES = [
    "portfolio_full_analysis",
    "portfolio_inline",
    "symbol_full_analysis",
    "symbol_inline",
    "crypto_full_analysis",
    "crypto_inline",
    "chat_messages",
]


@router.get("/usage", response_model=UsageOut)
def get_my_usage(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    from services.tier import get_user_plan, get_usage, get_limit

    plan = get_user_plan(user, db)
    items = []
    for feat in FEATURES:
        items.append(UsageLimitItem(
            feature=feat,
            used=get_usage(user.id, feat, plan),
            limit=get_limit(plan, feat),
        ))
    return UsageOut(plan=plan, usage=items)


# ─── Refund ────────────────────────────────────────────────────


class RefundEligibilityOut(BaseModel):
    eligible: bool
    days_left: Optional[int] = None
    reason: Optional[str] = None


@router.get("/refund-eligibility", response_model=RefundEligibilityOut)
def check_refund_eligibility(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    row = get_or_create_sub(db, user)

    if row.plan == "free" or row.status not in ("active", "trialing"):
        return RefundEligibilityOut(eligible=False, reason="No active paid subscription.")

    if row.has_used_refund:
        return RefundEligibilityOut(eligible=False, reason="Refund has already been used.")

    if not row.stripe_subscription_id:
        return RefundEligibilityOut(eligible=False, reason="No subscription found.")

    try:
        sub_obj = stripe.Subscription.retrieve(row.stripe_subscription_id)
        started_at = datetime.fromtimestamp(sub_obj["start_date"], tz=timezone.utc)
        age = datetime.now(timezone.utc) - started_at
        days_left = REFUND_WINDOW_DAYS - age.days

        if days_left <= 0:
            return RefundEligibilityOut(eligible=False, reason="Refund window has expired.")

        return RefundEligibilityOut(eligible=True, days_left=days_left)
    except Exception:
        logger.exception("refund_eligibility_check_failed user_id=%s", user.id)
        return RefundEligibilityOut(eligible=False, reason="Could not verify eligibility.")


class RefundOut(BaseModel):
    ok: bool
    message: str


@router.post("/refund", response_model=RefundOut)
def request_refund(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    row = get_or_create_sub(db, user)

    if row.plan == "free" or row.status not in ("active", "trialing"):
        raise HTTPException(400, detail="No active paid subscription to refund.")

    if row.has_used_refund:
        raise HTTPException(400, detail="You have already used your refund.")

    if not row.stripe_subscription_id:
        raise HTTPException(400, detail="No Stripe subscription found.")

    # Verify the refund window using Stripe as source of truth
    try:
        sub_obj = stripe.Subscription.retrieve(row.stripe_subscription_id)
    except Exception:
        logger.exception("stripe_sub_retrieve_failed user_id=%s", user.id)
        raise HTTPException(400, detail="Could not retrieve subscription details.")

    started_at = datetime.fromtimestamp(sub_obj["start_date"], tz=timezone.utc)
    age = datetime.now(timezone.utc) - started_at
    if age.days > REFUND_WINDOW_DAYS:
        raise HTTPException(
            400,
            detail=f"Refund window has expired. Refunds are available within {REFUND_WINDOW_DAYS} days of purchase.",
        )

    try:
        # Find the latest paid invoice with a charge
        invoices = stripe.Invoice.list(
            subscription=row.stripe_subscription_id,
            status="paid",
            limit=5,
        )

        charge_id = None
        payment_intent_id = None
        for inv in invoices.data:
            charge_id = inv.get("charge")
            payment_intent_id = inv.get("payment_intent")
            if charge_id or payment_intent_id:
                break

        logger.info(
            "refund_invoice_lookup user_id=%s invoices_found=%s charge=%s pi=%s",
            user.id, len(invoices.data), charge_id, payment_intent_id,
        )

        if not charge_id and not payment_intent_id:
            # No paid invoice — cancel without refund (nothing was charged)
            stripe.Subscription.cancel(row.stripe_subscription_id)
            logger.info("stripe_sub_canceled_no_charge user_id=%s", user.id)

            row.plan = "free"
            row.status = "canceled"
            row.has_used_refund = True
            row.refunded_at = datetime.now(timezone.utc)
            row.updated_at = datetime.now(timezone.utc)
            db.add(row)
            db.commit()

            return RefundOut(ok=True, message="Subscription canceled. No charge was found to refund.")

        refund_params: dict = {"reason": "requested_by_customer"}
        if charge_id:
            refund_params["charge"] = charge_id
        else:
            refund_params["payment_intent"] = payment_intent_id

        stripe.Refund.create(**refund_params)
        logger.info("stripe_refund_issued user_id=%s sub_id=%s", user.id, row.stripe_subscription_id)

        stripe.Subscription.cancel(row.stripe_subscription_id)
        logger.info("stripe_sub_canceled_for_refund user_id=%s", user.id)

        row.plan = "free"
        row.status = "canceled"
        row.has_used_refund = True
        row.refunded_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()

        return RefundOut(ok=True, message="Refund issued and subscription canceled.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("stripe_refund_failed user_id=%s", user.id)
        raise HTTPException(400, detail="Refund failed. Please contact support.")


# Webhook: no auth
@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not WEBHOOK_SECRET:
        raise HTTPException(500, detail="STRIPE_WEBHOOK_SECRET missing")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    if not sig:
        raise HTTPException(400, detail="Missing stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception as e:
        logger.exception("Stripe webhook signature verification failed")
        raise HTTPException(400, detail="Invalid webhook signature")

    if event["type"] in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ):
        sub_obj = event["data"]["object"]
        customer_id = sub_obj["customer"]
        stripe_sub_id = sub_obj["id"]
        status = sub_obj.get("status")

        # timestamps are unix seconds
        cpe = sub_obj.get("current_period_end")
        te = sub_obj.get("trial_end")

        # Determine plan by price id
        plan = "free"
        items = (sub_obj.get("items") or {}).get("data") or []
        price_id = items[0]["price"]["id"] if items else None
        if price_id == PRICE_PREMIUM:
            plan = "premium"
        elif price_id == PRICE_PRO:
            plan = "pro"

        logger.info("SUB EVENT customer_id: %s", customer_id)
        row = db.query(UserSubscription).filter_by(stripe_customer_id=customer_id).first()
        logger.info("DB lookup found: %s", bool(row))
        
        if row:
            row.stripe_subscription_id = stripe_sub_id
            row.status = status or row.status
            row.plan = plan
            row.current_period_end = datetime.fromtimestamp(cpe, tz=timezone.utc) if cpe else None
            row.trial_end = datetime.fromtimestamp(te, tz=timezone.utc) if te else None
            row.updated_at = datetime.now(timezone.utc)
            cap = sub_obj.get("cancel_at_period_end") or False
            cancel_at_ts = sub_obj.get("cancel_at")

            row.cancel_at_period_end = bool(cap)
            row.cancel_at = datetime.fromtimestamp(cancel_at_ts, tz=timezone.utc) if cancel_at_ts else None


            db.add(row)
            db.commit()
            logger.info("stripe_subscription_updated customer_id=%s plan=%s status=%s", customer_id, plan, status)

    logger.info("stripe_webhook_processed type=%s", event["type"])
    return {"received": True}
