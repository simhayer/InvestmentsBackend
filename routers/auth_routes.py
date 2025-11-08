# routers/auth_routes.py
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from models.user import User
from services.auth_service import get_current_user

import schemas.general as general
import services.crud as crud
from services.auth_service import (
    verify_password,
    create_access_token,
    decode_access_token,  # make sure this exists in your auth_service
)
from database import get_db

from pydantic import BaseModel, EmailStr
from services.auth_service import (
    get_password_hash,
    create_password_reset_token,
    decode_password_reset_token,
)
from urllib.parse import urlencode

router = APIRouter()

# --- Cookie config ---
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "auth_token")
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
AUTH_COOKIE_SAMESITE = "lax"
# AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax") == "lax"  # 'lax' for locahost, 'none' for prod
# Set to False only for local HTTP; must be True on HTTPS
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "true").lower() == "true"
AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN", None)  # e.g. ".yourdomain.com"

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

@router.post("/register")
def register(user: general.UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db, user.email, user.password)


# ✅ Login: set httpOnly cookie; do not return the token in the body
@router.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": str(user.id)})

    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite= "none" if AUTH_COOKIE_SAMESITE == "none" else "lax",
        max_age=AUTH_COOKIE_MAX_AGE,
        domain=AUTH_COOKIE_DOMAIN,
        path="/",
    )
    return resp


# Logout: clear cookie
@router.post("/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return resp

# Current user via cookie (SSR-friendly)
@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email}

class ForgotPasswordIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    token: str
    new_password: str

def _send_reset_email(to_email: str, reset_url: str):
    # Replace with your mailer; this is a stub so you can see it fast in logs.
    print(f"[MAIL] To: {to_email}\nReset your password: {reset_url}\n")

@router.post("/password/forgot")
def forgot_password(payload: ForgotPasswordIn, db: Session = Depends(get_db)):
    # Look up user silently
    user = crud.get_user_by_email(db, payload.email)
    if user:
        token = create_password_reset_token(user.id)
        qs = urlencode({"token": token})
        reset_url = f"{FRONTEND_URL}/reset-password?{qs}"
        _send_reset_email(payload.email, reset_url)
    # Always respond OK to prevent user enumeration
    return {"ok": True}

@router.post("/password/reset")
def reset_password(payload: ResetPasswordIn, db: Session = Depends(get_db)):
    data = decode_password_reset_token(payload.token)  # raises 401 if bad/expired
    sub = data.get("sub")
    user = db.get(User, int(sub)) if sub is not None else None
    if not user:
        # Also avoids saying whether the token/user is real
        raise HTTPException(status_code=400, detail="Invalid reset request")

    if len(payload.new_password) < 8:
        raise HTTPException(status_code=422, detail="Password too short")

    # Update password
    user.hashed_password = get_password_hash(payload.new_password)
    db.add(user)
    db.commit()

    # (Optional) Clear any existing auth cookie so user must sign in again
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(AUTH_COOKIE_NAME, path="/", domain=AUTH_COOKIE_DOMAIN)
    return resp