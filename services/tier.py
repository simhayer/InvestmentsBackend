# services/tier.py
"""
Tier / plan gating for WallStreetAI.

Usage in a route:
    from services.tier import require_tier, TierLimits

    @router.get("/full")
    async def get_full_analysis(
        ...,
        user=Depends(get_current_db_user),
        db: Session = Depends(get_db),
    ):
        require_tier(user, db, "portfolio_full_analysis")
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.user import User
from models.user_subscription import UserSubscription
from services.cache.cache_backend import cache_get, cache_set

logger = logging.getLogger(__name__)

# ─── Plan definitions ────────────────────────────────────────────
# Limits are per rolling window.  Default window = 24 h (86 400 s).
# Override per-feature in FEATURE_TTL_OVERRIDES.
# -1 = unlimited.

DAY_SEC = 86_400
WEEK_SEC = 7 * DAY_SEC

@dataclass
class PlanLimits:
    portfolio_full_analysis: int = 0
    portfolio_inline: int = 0
    symbol_full_analysis: int = 0
    symbol_inline: int = 0
    crypto_full_analysis: int = 0
    crypto_inline: int = 0
    connections: int = 0           # max brokerage connections


PLAN_LIMITS: Dict[str, PlanLimits] = {
    "free": PlanLimits(
        portfolio_full_analysis=1,    # 1 per WEEK (see TTL override below)
        portfolio_inline=5,
        symbol_full_analysis=3,       # 3 stocks per day
        symbol_inline=10,
        crypto_full_analysis=0,       # crypto is paid-only
        crypto_inline=0,
        connections=1,
    ),
    "premium": PlanLimits(
        portfolio_full_analysis=5,
        portfolio_inline=-1,
        symbol_full_analysis=15,
        symbol_inline=-1,
        crypto_full_analysis=5,
        crypto_inline=-1,
        connections=3,
    ),
    "pro": PlanLimits(
        portfolio_full_analysis=-1,
        portfolio_inline=-1,
        symbol_full_analysis=-1,
        symbol_inline=-1,
        crypto_full_analysis=-1,
        crypto_inline=-1,
        connections=-1,
    ),
}

# Features whose counters reset on a non-default window.
# Key = (plan, feature) or just feature (applies to all plans).
# Value = TTL in seconds.
FEATURE_TTL_OVERRIDES: Dict[str, int] = {
    "free:portfolio_full_analysis": WEEK_SEC,   # 1 per week on free
}


# ─── Helpers ─────────────────────────────────────────────────────

def get_user_plan(user: User, db: Session) -> str:
    """Return the active plan string for a user ('free' | 'premium' | 'pro')."""
    sub: Optional[UserSubscription] = (
        db.query(UserSubscription)
        .filter_by(user_id=user.id)
        .first()
    )
    if not sub:
        return "free"

    # Only count as paid if subscription is actually active or trialing
    if sub.status in ("active", "trialing"):
        return sub.plan or "free"

    return "free"


def _feature_ttl(plan: str, feature: str) -> int:
    """Return the counter TTL for a given plan + feature."""
    return FEATURE_TTL_OVERRIDES.get(f"{plan}:{feature}", DAY_SEC)


def _usage_key(user_id: int, feature: str, plan: str = "free") -> str:
    """Redis key for the usage counter.  Includes the TTL window so
    weekly and daily counters don't collide."""
    ttl = _feature_ttl(plan, feature)
    window = "w" if ttl > DAY_SEC else "d"
    return f"tier:usage:{window}:{user_id}:{feature}"


def get_usage(user_id: int, feature: str, plan: str = "free") -> int:
    """How many times this user has used *feature* in the current window."""
    raw = cache_get(_usage_key(user_id, feature, plan))
    if raw is None:
        return 0
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def increment_usage(user_id: int, feature: str, plan: str = "free") -> int:
    """Bump the counter and return the new value."""
    key = _usage_key(user_id, feature, plan)
    current = get_usage(user_id, feature, plan)
    new_val = current + 1
    ttl = _feature_ttl(plan, feature)
    cache_set(key, new_val, ttl_seconds=ttl)
    return new_val


def get_limit(plan: str, feature: str) -> int:
    """Return the limit for a plan + feature.  -1 = unlimited."""
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    return getattr(limits, feature, 0)


# ─── Public gate ─────────────────────────────────────────────────

def require_tier(
    user: User,
    db: Session,
    feature: str,
    *,
    increment: bool = True,
) -> str:
    """
    Check that *user* hasn't exceeded the limit for *feature*.

    Raises 403 with a JSON body the frontend can parse:
        { "detail": "...", "code": "TIER_LIMIT", "plan": "free",
          "feature": "...", "limit": 3, "used": 3 }

    Returns the plan string for convenience.
    """
    plan = get_user_plan(user, db)
    limit = get_limit(plan, feature)

    # -1 means unlimited
    if limit == -1:
        if increment:
            increment_usage(user.id, feature, plan)
        return plan

    # 0 means feature is entirely unavailable on this plan
    used = get_usage(user.id, feature, plan)

    if limit == 0 or used >= limit:
        ttl = _feature_ttl(plan, feature)
        window_label = "weekly" if ttl > DAY_SEC else "daily"
        raise HTTPException(
            status_code=403,
            detail={
                "message": f"You've reached your {window_label} limit for this feature. Upgrade your plan to continue.",
                "code": "TIER_LIMIT",
                "plan": plan,
                "feature": feature,
                "limit": limit,
                "used": used,
            },
        )

    if increment:
        increment_usage(user.id, feature, plan)

    return plan
