# services/supabase_auth.py
import os
import time
import httpx
from jose import jwt, JWTError
from fastapi import HTTPException, Request, status

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from models.user import User
from database import get_db

SUPABASE_PROJECT_URL = os.getenv("SUPABASE_PROJECT_URL", "").rstrip("/")
SUPABASE_JWT_AUD = os.getenv("SUPABASE_JWT_AUD", "authenticated")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")  # <-- add this
if not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_JWT_SECRET is not set in environment variables")

# Simple in-memory JWKS cache
_JWKS_CACHE = {"keys": None, "ts": 0}
_JWKS_TTL_SECONDS = 60 * 60  # 1 hour


def _get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return auth.split(" ", 1)[1].strip()


async def _get_jwks() -> dict:
    now = int(time.time())
    if _JWKS_CACHE["keys"] and (now - _JWKS_CACHE["ts"] < _JWKS_TTL_SECONDS):
        return _JWKS_CACHE["keys"]

    if not SUPABASE_PROJECT_URL:
        raise RuntimeError("SUPABASE_PROJECT_URL is not set")

    jwks_url = f"{SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(jwks_url)
        r.raise_for_status()
        jwks = r.json()

    _JWKS_CACHE["keys"] = jwks
    _JWKS_CACHE["ts"] = now
    return jwks


async def get_current_supabase_user(request: Request) -> dict:
    token = _get_bearer_token(request)
    
    if not SUPABASE_JWT_SECRET:
        raise RuntimeError("SUPABASE_JWT_SECRET is not configured")

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,                 # HS256 uses shared secret
            algorithms=["HS256"],
            audience=SUPABASE_JWT_AUD,
            issuer=f"{SUPABASE_PROJECT_URL}/auth/v1",
        )
        return payload
    except JWTError as e:
        print(f"❌ JWTError: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")

def get_current_db_user(
    db: Session = Depends(get_db),
    payload: dict = Depends(get_current_supabase_user),
) -> User:
    # 1) Read Supabase UUID from JWT
    supabase_user_id = payload.get("sub")
    if not supabase_user_id:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    # 2) Try to find existing local user
    user = (
        db.query(User)
        .filter(User.supabase_user_id == str(supabase_user_id))
        .first()
    )
    if user:
        return user

    # 3) Auto-create local user on first login
    # Email can be in different places depending on Supabase config
    email = payload.get("email")
    if not email:
        email = (payload.get("user_metadata") or {}).get("email")

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Cannot create user: email missing from Supabase token",
        )

    user = User(
        email=email,
        supabase_user_id=str(supabase_user_id),
        # hashed_password intentionally omitted (Supabase manages auth)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user