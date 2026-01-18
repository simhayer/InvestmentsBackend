# main.py
import os
from dotenv import load_dotenv
load_dotenv()
# if os.getenv("DEBUGPY", "0") == "1":
#     import debugpy
#     debugpy.listen(("0.0.0.0", 5678))
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.user_routes import router as user_router
from routers.holdings_routes import router as holdings_router
from routers.finnhub_routes import router as finnhub_router  # keep this as-is
from routers.plaid_routes import router as plaid_routers
from routers.ai_routes import router as ai_router  # keep this as-is
from routers.investment_routes import router as investment_router
from routers.portfolio_routes import router as portfolio_router
from routers.news_routes import router as news_router
from routers.marktet_routes import router as market_router
from routers.onboarding_routes import router as onboarding_router
from routers.billing_routes import router as billing_router
from routers.crypto_routes import router as crypto_router
from routers.ai_chat_routes import router as ai_chat_router
from routers.v2.analyse_symbol_routes import router as analyse_symbol_router
from routers.filing_routes import router as filing_router 

# load crypto catalog on startup
from contextlib import asynccontextmanager
from database import SessionLocal
from services.binance_service import refresh_crypto_catalog, load_crypto_catalog

# db startup
from database import Base, engine
import models  # this triggers models/__init__.py which imports all tables

Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print("[startup] create_all failed:", repr(e))

    try:
        print("[startup] loading crypto catalog")
        db = SessionLocal()
        try:
            # await refresh_crypto_catalog(db, provider="binance")
            load_crypto_catalog(db, provider="binance")
        finally:
            db.close()
    except Exception as e:
        print("[startup] crypto catalog load failed:", repr(e))

    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://investments-backend-1db4.vercel.app",
    "https://investments-backend-seven.vercel.app",
    "https://investmentai.life",
    "https://www.wallstreetai.io",
    "https://wallstreetai.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

# Include routers
app.include_router(user_router)
app.include_router(holdings_router)
app.include_router(finnhub_router, prefix="/api/finnhub")
app.include_router(plaid_routers, prefix="/api/plaid")
app.include_router(ai_router, prefix="/api/ai")
app.include_router(investment_router, prefix="/api/investment")
app.include_router(portfolio_router, prefix="/api/portfolio")
app.include_router(news_router, prefix="/api/news")
app.include_router(market_router, prefix="/api/market")
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])
app.include_router(onboarding_router, prefix="/api/onboarding")
app.include_router(crypto_router, prefix="/api/crypto", tags=["crypto"])
app.include_router(ai_chat_router, prefix="/api/ai", tags=["ai-chat"])
app.include_router(analyse_symbol_router, prefix="/api/v2/ai", tags=["v2-ai"])
app.include_router(filing_router, prefix="/api/filings", tags=["filings"])
