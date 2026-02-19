# main.py
import logging
import os
import json
import tempfile
from dotenv import load_dotenv
load_dotenv()

_sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if _sa_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _tmp.write(_sa_json)
    _tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

# if os.getenv("DEBUGPY", "0") == "1":
#     import debugpy
#     debugpy.listen(("0.0.0.0", 5678)
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from middleware.rate_limit import limiter

logger = logging.getLogger(__name__)
from routers.user_routes import router as user_router
from routers.holdings_routes import router as holdings_router
from routers.finnhub_routes import router as finnhub_router  # keep this as-is
from routers.plaid_routes import router as plaid_routers
from routers.ai_routes import router as ai_router  # keep this as-is
from routers.ai_chat_routes import router as ai_chat_router
from routers.investment_routes import router as investment_router
from routers.portfolio_routes import router as portfolio_router
from routers.news_routes import router as news_router
from routers.marktet_routes import router as market_router
from routers.onboarding_routes import router as onboarding_router
from routers.billing_routes import router as billing_router
from routers.crypto_routes import router as crypto_router
from routers.filing_routes import router as filing_router
from routers.v2.analyze_symbol_routes import router as analyze_symbol_router
from routers.v2.analyze_portfolio_routes import router as analyze_portfolio_router
from routers.v2.analyze_crypto_routes import router as analyze_crypto_router


# load crypto catalog on startup
from contextlib import asynccontextmanager
from database import SessionLocal
from services.binance_service import refresh_crypto_catalog, load_crypto_catalog

# db startup — models are imported so Alembic autogenerate can see them.
# Migrations are managed by Alembic; no create_all() on startup.
from database import Base, engine  # noqa: F401
import models  # noqa: F401  — triggers models/__init__.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("[startup] loading crypto catalog")
        db = SessionLocal()
        try:
            # await refresh_crypto_catalog(db, provider="binance")
            load_crypto_catalog(db, provider="binance")
        finally:
            db.close()
    except Exception as e:
        logger.exception("[startup] crypto catalog load failed: %s", repr(e))

    yield

app = FastAPI(lifespan=lifespan)

# ─── Rate limiting ──────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS ────────────────────────────────────────────────────────
_DEFAULT_ORIGINS = (
    "http://localhost:3000,"
    "http://localhost:8000,"
    "https://investments-backend-1db4.vercel.app,"
    "https://investments-backend-seven.vercel.app,"
    "https://investmentai.life,"
    "https://www.wallstreetai.io,"
    "https://wallstreetai.io"
)
origins = [
    o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

# ─── Health check ────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health_check():
    return JSONResponse({"status": "ok"})

# Include routers
app.include_router(user_router)
app.include_router(holdings_router)
app.include_router(finnhub_router, prefix="/api/finnhub")
app.include_router(plaid_routers, prefix="/api/plaid")
app.include_router(ai_router, prefix="/api/ai")
app.include_router(ai_chat_router, prefix="/api/ai", tags=["ai-chat"])
app.include_router(investment_router, prefix="/api/investment")
app.include_router(portfolio_router, prefix="/api/portfolio")
app.include_router(news_router, prefix="/api/news")
app.include_router(market_router, prefix="/api/market")
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])
app.include_router(onboarding_router, prefix="/api/onboarding")
app.include_router(crypto_router, prefix="/api/crypto", tags=["crypto"])
app.include_router(filing_router, prefix="/api/filings", tags=["filings"])
app.include_router(analyze_symbol_router, prefix="/api/analyze/symbol", tags=["analyze-symbol"])
app.include_router(analyze_portfolio_router, prefix="/api/portfolio/analysis", tags=["analyze-portfolio"])
app.include_router(analyze_crypto_router, prefix="/api/analyze/crypto", tags=["analyze-crypto"])
