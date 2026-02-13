# database.py
import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the environment")

# ─── Connection-pool tuning ────────────────────────────────────────
# These defaults are sensible for a Railway / small-VPS deployment.
# Override via env vars for larger setups.
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))           # steady-state connections
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))     # burst above pool_size
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))     # seconds to wait for a conn
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))   # recycle every 30 min (avoids stale PG conns)

engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=True,  # test connection liveness before checkout
)

logger.info(
    "DB pool configured: size=%d, max_overflow=%d, recycle=%ds, pre_ping=True",
    POOL_SIZE, MAX_OVERFLOW, POOL_RECYCLE,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
