# main.py
import os
if os.getenv("DEBUGPY", "0") == "1":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth_routes import router as auth_router
from routers.holdings_routes import router as holdings_router
from routers.finnhub_routes import router as finnhub_router  # keep this as-is
from routers.plaid_routes import router as plaid_routers
from routers.ai_routes import router as ai_router  # keep this as-is
from routers.investment_routes import router as investment_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(holdings_router)
app.include_router(finnhub_router, prefix="/api/finnhub")
app.include_router(plaid_routers, prefix="/api/plaid")
app.include_router(ai_router, prefix="/api/ai")
app.include_router(investment_router, prefix="/api/market")

# db startup
from database import Base, engine
import models  # this triggers models/__init__.py which imports all tables

Base.metadata.create_all(bind=engine)