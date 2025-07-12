# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import models
from database import engine
from routers.auth_routes import router as auth_router
from routers.holdings_routes import router as holdings_router
from routers.finnhub_routes import router as finnhub_router  # keep this as-is

models.Base.metadata.create_all(bind=engine)

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
