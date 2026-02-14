"""Assembles a multi-layer system prompt for the chat agent.

Layers:
  1. Base persona
  2. Tool manifest
  3. Page context (from frontend)
  4. Investor profile (loaded server-side from UserOnboardingProfile)
  5. Behavioral rules (page-aware references, portfolio guardrails)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from services.ai.chat.chat_models import ChatContext, ChatRequest
from services.ai.chat.chat_prompts import build_finance_system_prompt, build_tool_manifest_prompt

logger = logging.getLogger(__name__)


def _load_investor_profile(db: Session, user_id: int) -> Optional[Dict[str, Any]]:
    """Load the investor profile from onboarding — mirrors the pattern in
    ``analyze_portfolio_aggregator.py`` (lightweight single-row query)."""
    try:
        from models.user_onboarding_profile import UserOnboardingProfile

        profile = (
            db.query(UserOnboardingProfile)
            .filter(UserOnboardingProfile.user_id == user_id)
            .first()
        )
        if profile and profile.completed_at:
            return {
                "risk_level": profile.risk_level,
                "time_horizon": profile.time_horizon,
                "primary_goal": profile.primary_goal,
                "experience_level": profile.experience_level,
                "age_band": profile.age_band,
                "country": profile.country,
                "asset_preferences": profile.asset_preferences,
                "style_preference": profile.style_preference,
            }
    except Exception:
        logger.warning("context_assembler: failed to load investor profile", exc_info=True)
    return None


# ── Public API ──────────────────────────────────────────────────────────

def build_system_prompt(
    *,
    db: Session,
    user_id: int,
    req: ChatRequest,
) -> str:
    """Build the full multi-layer system prompt."""
    parts: list[str] = []

    # Layer 1 — base persona
    parts.append(build_finance_system_prompt().strip())

    # Layer 2 — tool manifest
    parts.append(build_tool_manifest_prompt().strip())

    # Layer 3 — page context
    page = req.context.page if req.context else None
    if page:
        lines = ["\n## Current Page Context"]
        lines.append(f"The user is currently on the **{page.page_type}** page (route: {page.route}).")
        if page.symbol:
            lines.append(f'Active symbol: {page.symbol}.')
        if page.summary:
            lines.append(f"Visible data summary: {page.summary}")
        if page.data_snapshot:
            lines.append(f"Data snapshot: {json.dumps(page.data_snapshot, ensure_ascii=True)}")
        parts.append("\n".join(lines))

    # Layer 4 — investor profile (server-side)
    investor = _load_investor_profile(db, user_id)
    if investor:
        lines = ["\n## Investor Profile"]
        if investor.get("risk_level"):
            lines.append(f"- Risk tolerance: {investor['risk_level']}")
        if investor.get("time_horizon"):
            lines.append(f"- Time horizon: {investor['time_horizon']}")
        if investor.get("primary_goal"):
            lines.append(f"- Primary goal: {investor['primary_goal']}")
        if investor.get("experience_level"):
            lines.append(f"- Experience level: {investor['experience_level']}")
        if investor.get("country"):
            lines.append(f"- Country: {investor['country']}")
        if investor.get("style_preference"):
            lines.append(f"- Style: {investor['style_preference']}")
        if investor.get("asset_preferences"):
            prefs = investor["asset_preferences"]
            enabled = [k for k, v in prefs.items() if v]
            if enabled:
                lines.append(f"- Preferred assets: {', '.join(enabled)}")
        lines.append(
            "\nUse this profile to calibrate the complexity and tone of your "
            "answers.  A beginner needs simpler language; an advanced user "
            "appreciates deeper quantitative detail."
        )
        parts.append("\n".join(lines))

    # Layer 5 — behavioral rules
    rules = ["\n## Behavioral Rules"]
    if page and page.symbol:
        rules.append(
            '- When the user says "this stock", "this chart", or "this company", '
            f"they are referring to **{page.symbol}** from the page context."
        )
    if page and page.page_type == "dashboard":
        rules.append(
            "- The user is viewing their portfolio dashboard. Tailor answers to "
            "their own holdings and portfolio performance."
        )
    if page and page.page_type == "holdings":
        rules.append(
            "- The user is looking at their holdings list. If they ask 'which of my '... "
            "questions, reference their holdings data."
        )
    rules.append(
        "- When discussing the user's portfolio, always fetch latest data via "
        "tools rather than relying on stale context alone."
    )
    rules.append(
        "- Never fabricate numbers. If data is unavailable, state it explicitly."
    )
    parts.append("\n".join(rules))

    return "\n\n".join(parts)


def build_context_text(req: ChatRequest) -> str:
    """Serialize request context (excluding page, which goes into system prompt)
    into a compact string for the user prompt."""
    if not req.context:
        return ""
    data = req.context.model_dump(exclude_none=True, exclude={"page"})
    if not data:
        return ""
    return json.dumps(data, ensure_ascii=True)
