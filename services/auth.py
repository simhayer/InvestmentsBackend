# services/auth.py
from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Cookie, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import get_db
from models.user import User

# ========================
# Config
# ========================

# Load from env in prod; fall back only for local dev
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Name of the cookie we set in the auth router
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "auth_token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Keep bearer support for API clients
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ========================
# Password helpers
# ========================

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ========================
# JWT helpers
# ========================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode & verify JWT. Raises HTTPException(401) on failure.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ========================
# User dependencies
# ========================

def _get_user_by_sub(db: Session, sub: str) -> User:
    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token subject")
    user = db.get(User, user_id)  # preferred over deprecated query().get()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def get_current_user_from_bearer(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_access_token(token)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return _get_user_by_sub(db, sub)

def get_current_user_from_cookie(
    db: Session = Depends(get_db),
    auth_token: Optional[str] = Cookie(default=None, alias=AUTH_COOKIE_NAME),
) -> User:
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(auth_token)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return _get_user_by_sub(db, sub)
