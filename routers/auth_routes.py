# routers/auth_routes.py
import os
from fastapi import APIRouter, Depends, HTTPException, Cookie
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

router = APIRouter()

# --- Cookie config ---
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "auth_token")
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
AUTH_COOKIE_SAMESITE = "none"  # 'lax' for locahost, 'none' for prod
# Set to False only for local HTTP; must be True on HTTPS
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "true").lower() == "true"
AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN", None)  # e.g. ".yourdomain.com"


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
        samesite=AUTH_COOKIE_SAMESITE,
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
