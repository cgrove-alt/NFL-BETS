"""
FastAPI application for NFL Bets API.

Main entry point for the REST API that exposes:
- Health check endpoints
- Value bet discovery
- Bankroll and bet tracking
- Model status monitoring
- Scheduler job control
- Analytics and performance metrics

Run with:
    uvicorn api.main:app --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, value_bets, bankroll, models, jobs, analytics, games
from api.state import AppState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes components on startup and cleans up on shutdown.
    """
    logger.info("Starting NFL Bets API...")

    # Initialize application state
    state = AppState()
    await state.initialize()
    app.state.app_state = state

    logger.info("NFL Bets API started successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down NFL Bets API...")
    await state.shutdown()
    logger.info("NFL Bets API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="NFL Bets API",
    description="Real-time NFL betting prediction and value detection system",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Configure CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local Next.js dev
        "https://*.vercel.app",   # Vercel preview deployments
        # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(value_bets.router, prefix="/api", tags=["Value Bets"])
app.include_router(bankroll.router, prefix="/api", tags=["Bankroll"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(games.router, prefix="/api", tags=["Games"])


@app.get("/")
async def root():
    """Root endpoint redirects to API documentation."""
    return {
        "name": "NFL Bets API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/api/health",
    }
