# routes/onboarding_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models.user import User
from models.user_onboarding_profile import UserOnboardingProfile
from services.supabase_auth import get_current_db_user

router = APIRouter()

class OnboardingOut(BaseModel):
    completed: bool
    current_step: int
    time_horizon: str | None = None
    primary_goal: str | None = None
    risk_level: str | None = None
    experience_level: str | None = None
    age_band: str | None = None
    country: str | None = None
    asset_preferences: dict | None = None
    style_preference: str | None = None
    notification_level: str | None = None
    notes: str | None = None

class OnboardingUpdate(BaseModel):
    current_step: int | None = None
    time_horizon: str | None = None
    primary_goal: str | None = None
    risk_level: str | None = None
    experience_level: str | None = None
    age_band: str | None = None
    country: str | None = None
    asset_preferences: dict | None = None
    style_preference: str | None = None
    notification_level: str | None = None
    notes: str | None = None

def get_or_create_profile(db: Session, user: User) -> UserOnboardingProfile:
    profile = (
        db.query(UserOnboardingProfile)
        .filter(UserOnboardingProfile.user_id == user.id)
        .first()
    )
    if profile:
        return profile

    profile = UserOnboardingProfile(user_id=user.id, current_step=0)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile

@router.get("", response_model=OnboardingOut)
def get_onboarding(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_db_user),
):
    profile = get_or_create_profile(db, current_user)
    return OnboardingOut(
        completed=profile.completed_at is not None,
        current_step=profile.current_step,
        time_horizon=profile.time_horizon,
        primary_goal=profile.primary_goal,
        risk_level=profile.risk_level,
        experience_level=profile.experience_level,
        age_band=profile.age_band,
        country=profile.country,
        asset_preferences=profile.asset_preferences,
        style_preference=profile.style_preference,
        notification_level=profile.notification_level,
        notes=profile.notes,
    )

@router.patch("", response_model=OnboardingOut)
def update_onboarding(
    payload: OnboardingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_db_user),
):
    try:
        profile = get_or_create_profile(db, current_user)

        # Update only fields provided
        data = payload.model_dump(exclude_unset=True)
        for k, v in data.items():
            setattr(profile, k, v)

        db.add(profile)
        db.commit()
        db.refresh(profile)

        return OnboardingOut(
            completed=profile.completed_at is not None,
            current_step=profile.current_step,
            time_horizon=profile.time_horizon,
            primary_goal=profile.primary_goal,
            risk_level=profile.risk_level,
            experience_level=profile.experience_level,
            age_band=profile.age_band,
            country=profile.country,
            asset_preferences=profile.asset_preferences,
            style_preference=profile.style_preference,
            notification_level=profile.notification_level,
            notes=profile.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update onboarding profile: {e}")

@router.post("/complete", response_model=OnboardingOut)
def complete_onboarding(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_db_user),
):
    profile = get_or_create_profile(db, current_user)
    if profile.completed_at is not None:
        # already completed
        return OnboardingOut(
            completed=True,
            current_step=profile.current_step,
            time_horizon=profile.time_horizon,
            primary_goal=profile.primary_goal,
            risk_level=profile.risk_level,
            experience_level=profile.experience_level,
            age_band=profile.age_band,
            country=profile.country,
            asset_preferences=profile.asset_preferences,
            style_preference=profile.style_preference,
            notification_level=profile.notification_level,
            notes=profile.notes,
        )

    # profile.completed_at = datetime.now(timezone.utc)
    profile.completed_at = func.now()
    db.add(profile)
    db.commit()
    db.refresh(profile)

    return OnboardingOut(
        completed=True,
        current_step=profile.current_step,
        time_horizon=profile.time_horizon,
        primary_goal=profile.primary_goal,
        risk_level=profile.risk_level,
        experience_level=profile.experience_level,
        age_band=profile.age_band,
        country=profile.country,
        asset_preferences=profile.asset_preferences,
        style_preference=profile.style_preference,
        notification_level=profile.notification_level,
        notes=profile.notes,
    )
